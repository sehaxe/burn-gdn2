use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Distribution as TensorDistribution, Tensor};

use burn::tensor::activation::{sigmoid, silu, softplus};

use crate::config::{Gdn2Config, Gdn2Mode};
use crate::forward::chunk_wy_forward;
use crate::kernel::fused_recurrent::fused_recurrent_gdn2;
use crate::l2norm::l2_normalize;
use crate::short_conv::short_conv_1d;

/// GDN-2 (Gated DeltaNet 2) token-mixing layer.
///
/// GDN-2 decouples the erase and write operations of the gated delta rule
/// into two independent channel-wise gates:
///   - `b` (erase gate, `[0,1]^{d_k}`) — how much of each key channel to erase
///   - `w` (write gate, `[0,1]^{d_v}`) — how much of each value channel to write
///
/// The per-token recurrence on the matrix state `S ∈ R^{d_k × d_v}` is:
///
///   S_t = (I - k_t (b_t ⊙ k_t)ᵀ) diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)ᵀ
///
/// Two forward paths are available:
/// - `FusedRecurrent`: token-by-token scan (inference, state-passing)
/// - `Chunk`: chunked WY representation (training, cuBLAS-friendly matmuls)
#[derive(Module, Debug)]
pub struct GatedDeltaNet2<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub f_proj_0: Linear<B>,
    pub f_proj_1: Linear<B>,
    pub b_proj: Linear<B>,
    pub w_proj: Linear<B>,
    pub g_proj_0: Linear<B>,
    pub g_proj_1: Linear<B>,
    pub a_log: Param<Tensor<B, 1>>,
    pub dt_bias: Param<Tensor<B, 1>>,
    pub o_norm_weight: Param<Tensor<B, 1>>,
    pub o_proj: Linear<B>,
    pub q_conv_w: Param<Tensor<B, 2>>,
    pub k_conv_w: Param<Tensor<B, 2>>,
    pub v_conv_w: Param<Tensor<B, 2>>,
    pub config: Gdn2Config,
    pub decay_factors: Option<Param<Tensor<B, 2>>>,
}

impl<B: Backend> GatedDeltaNet2<B> {
    /// Create a new GDN-2 layer from a configuration.
    pub fn new(cfg: &Gdn2Config, device: &B::Device) -> Self {
        let d = cfg.hidden_size;
        let h = cfg.num_heads;
        let hk = cfg.head_dim;
        let hv = cfg.num_v_heads.unwrap_or(h);
        let kd = h * hk;
        let v_head = (hk as f32 * cfg.expand_v) as usize;
        let vd = hv * v_head;
        let gain = 2.0f64.powf(-2.5);

        let lin = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        let lin_b = |i, o| LinearConfig::new(i, o).with_bias(true).init(device);

        let rand_w = |c: usize| -> Param<Tensor<B, 2>> {
            let w = Tensor::<B, 2>::random(
                [c, 4],
                TensorDistribution::Uniform(
                    -gain * (1.0f64 / c as f64).sqrt(),
                    gain * (1.0f64 / c as f64).sqrt(),
                ),
                device,
            );
            Param::from_tensor(w)
        };

        let a_init = Tensor::<B, 1>::random(
            [h],
            TensorDistribution::Uniform(1.0f64.ln(), 16.0f64.ln()),
            device,
        );

        let dt = {
            let raw = Tensor::<B, 1>::random(
                [kd],
                TensorDistribution::Uniform(0.001f64.ln(), 0.1f64.ln()),
                device,
            )
            .exp()
            .clamp(1e-4, f32::MAX);
            raw.clone() + (-raw.clone()).exp().neg().add_scalar(1.0).log()
        };

        Self {
            q_proj: lin(d, kd),
            k_proj: lin(d, kd),
            v_proj: lin(d, vd),
            f_proj_0: lin(d, v_head),
            f_proj_1: lin(v_head, kd),
            b_proj: lin(d, kd),
            w_proj: lin(d, vd),
            g_proj_0: lin(d, v_head),
            g_proj_1: lin_b(v_head, vd),
            a_log: Param::from_tensor(a_init),
            dt_bias: Param::from_tensor(dt),
            o_norm_weight: Param::from_tensor(Tensor::ones([v_head], device)),
            o_proj: lin(vd, d),
            q_conv_w: rand_w(kd),
            k_conv_w: rand_w(kd),
            v_conv_w: rand_w(vd),
            decay_factors: cfg
                .min_decay
                .map(|_| Param::from_tensor(Tensor::ones([h, hk], device).mul_scalar(2.0))),
            config: cfg.clone(),
        }
    }

    /// Validate input dimensions match the model configuration.
    ///
    /// # Panics
    ///
    /// Panics if `hidden_states` last dimension does not equal `config.hidden_size`,
    /// or if `state` is provided with mismatched dimensions.
    #[inline]
    fn validate(&self, hidden_states: &Tensor<B, 3>) {
        let d = hidden_states.shape().dims::<3>()[2];
        assert_eq!(
            d, self.config.hidden_size,
            "GatedDeltaNet2: expected hidden_states with last dim {} but got {}. \
             Check that the model hidden_size matches your input.",
            self.config.hidden_size, d,
        );
    }

    /// Inference forward: one sequence pass with state management.
    ///
    /// When `update_state` is true, the recurrent state is updated (normal
    /// autoregressive decoding). When false, the state is read without
    /// modification (useful for prefill).
    ///
    /// # Panics
    ///
    /// Panics if `hidden_states` last dimension does not equal `config.hidden_size`.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        state: &mut Option<Tensor<B, 4>>,
        update_state: bool,
    ) -> Tensor<B, 3> {
        self.validate(&hidden_states);
        let [batch, tokens, _] = hidden_states.shape().dims::<3>();
        let projected = self.project(hidden_states);
        let hk = self.config.head_dim;
        let hv = projected.hv;
        let vd = projected.vd;
        let scale = (hk as f64).powf(-0.5);

        let (output, new_state): (Tensor<B, 4>, Option<Tensor<B, 4>>) = if update_state {
            let (o, s) = fused_recurrent_gdn2(
                projected.q,
                projected.k,
                projected.v,
                projected.g,
                projected.b,
                projected.w,
                state.take(),
                scale,
                true,
            );
            (o, s)
        } else {
            let mem = state
                .clone()
                .unwrap_or_else(|| Tensor::zeros([batch, hv, hk, vd / hv], &projected.q.device()));
            let out = projected
                .q
                .matmul(mem.clone())
                .mul_scalar(scale)
                .permute([0, 2, 1, 3]);
            (out, Some(mem))
        };
        *state = new_state;

        let out_4d = output.permute([0, 2, 1, 3]);
        let out_norm =
            rms_norm_gate_per_head(out_4d, projected.gate, projected.o_norm, projected.eps);
        projected
            .o_proj
            .forward(out_norm.reshape([batch, tokens, vd]))
    }

    /// Training forward: full-sequence forward, selects mode based on config.
    ///
    /// `FusedRecurrent` mode: token-by-token scan for small models.
    /// `Chunk` mode: chunked WY representation for training at scale.
    ///
    /// # Panics
    ///
    /// Panics if `hidden_states` last dimension does not equal `config.hidden_size`.
    pub fn forward_train(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        self.validate(&hidden_states);
        let [batch, tokens, _] = hidden_states.shape().dims::<3>();
        let projected = self.project(hidden_states);
        let hk = self.config.head_dim;
        let hv = projected.hv;
        let vd = projected.vd;
        let scale = (hk as f64).powf(-0.5);
        let device = projected.q.device();
        let state = Tensor::zeros([batch, hv, hk, vd / hv], &device);

        let output = match self.config.mode {
            Gdn2Mode::FusedRecurrent => {
                let (o, _s) = fused_recurrent_gdn2(
                    projected.q,
                    projected.k,
                    projected.v,
                    projected.g,
                    projected.b,
                    projected.w,
                    Some(state),
                    scale,
                    true,
                );
                o
            }
            Gdn2Mode::Chunk => {
                let (o, _s) = chunk_wy_forward(
                    projected.q,
                    projected.k,
                    projected.v,
                    projected.g,
                    projected.b,
                    projected.w,
                    state,
                    scale,
                    self.config.chunk_size,
                );
                o
            }
        };

        let out_4d = output.permute([0, 2, 1, 3]);
        let out_norm =
            rms_norm_gate_per_head(out_4d, projected.gate, projected.o_norm, projected.eps);
        projected
            .o_proj
            .forward(out_norm.reshape([batch, tokens, vd]))
    }

    /// Shared projection pipeline: input → Q/K/V/B/W + gates.
    pub fn project(&self, hidden_states: Tensor<B, 3>) -> ProjectedInputs<'_, B> {
        let [batch, tokens, _] = hidden_states.shape().dims::<3>();
        let h = self.config.num_heads;
        let hk = self.config.head_dim;
        let hv = self.config.num_v_heads.unwrap_or(h);
        let v_head = (hk as f32 * self.config.expand_v) as usize;
        let vd = hv * v_head;
        let kd = h * hk;
        let use_sc = self.config.use_short_conv;

        let to_4d = |t: Tensor<B, 3>, n: usize, d: usize| -> Tensor<B, 4> {
            let [b, tt, _] = t.shape().dims::<3>();
            t.reshape([b, tt, n, d]).permute([0, 2, 1, 3])
        };

        let q_raw = self.q_proj.forward(hidden_states.clone());
        let k_raw = self.k_proj.forward(hidden_states.clone());
        let v_raw = self.v_proj.forward(hidden_states.clone());

        let q_act = if use_sc {
            short_conv_1d(q_raw, self.q_conv_w.val())
        } else {
            silu(q_raw)
        };
        let k_act = if use_sc {
            short_conv_1d(k_raw, self.k_conv_w.val())
        } else {
            silu(k_raw)
        };
        let v_act = if use_sc {
            short_conv_1d(v_raw, self.v_conv_w.val())
        } else {
            silu(v_raw)
        };

        let f_hid = self.f_proj_0.forward(hidden_states.clone());
        let f_out = self.f_proj_1.forward(f_hid);
        let dt_b = self.dt_bias.val().reshape([1, 1, kd]);
        let a_exp = self
            .a_log
            .val()
            .exp()
            .reshape([1, h, 1])
            .repeat(&[1, 1, hk])
            .reshape([1, 1, kd]);
        let g_pre = softplus(f_out + dt_b, 1.0);
        let g = -a_exp * g_pre;

        let g = if let (Some(df), Some(min_d)) = (&self.decay_factors, self.config.min_decay) {
            let alpha = (1.0 - min_d) as f32;
            let factor = sigmoid(df.val())
                .mul_scalar(alpha)
                .reshape([1, h, hk])
                .reshape([1, 1, kd]);
            let d_lower_bounded = factor.mul(g.exp()).add_scalar(min_d as f32);
            d_lower_bounded.log()
        } else {
            g
        };

        let b_gate = sigmoid(self.b_proj.forward(hidden_states.clone()));
        let w_gate = sigmoid(self.w_proj.forward(hidden_states.clone()));

        // L2-normalize q and k along the head dimension
        let q_norm = l2_normalize(q_act, 1e-6);
        let k_norm = l2_normalize(k_act, 1e-6);

        let mut q_4d = to_4d(q_norm, h, hk);
        let mut k_4d = to_4d(k_norm, h, hk);
        let v_4d = to_4d(v_act, hv, v_head);
        let mut g_4d = to_4d(g, h, hk);
        let mut b_4d = to_4d(b_gate, h, hk);
        let w_4d = to_4d(w_gate, hv, v_head);

        // Repeat key-side tensors for grouped value attention (GVA)
        if hv > h {
            let rep = hv / h;
            let r = |t: Tensor<B, 4>| -> Tensor<B, 4> {
                t.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, rep, 1])
                    .reshape([batch, hv, tokens, hk])
            };
            q_4d = r(q_4d);
            k_4d = r(k_4d);
            g_4d = r(g_4d);
            b_4d = r(b_4d);
        }

        if self.config.allow_neg_eigval {
            b_4d = b_4d.mul_scalar(2.0);
        }

        let g_hid = self.g_proj_0.forward(hidden_states);
        let gate_signal = self.g_proj_1.forward(g_hid);
        let gate_4d = gate_signal.reshape([batch, tokens, hv, v_head]);

        ProjectedInputs {
            q: q_4d,
            k: k_4d,
            v: v_4d,
            g: g_4d,
            b: b_4d,
            w: w_4d,
            gate: gate_4d,
            o_norm: self.o_norm_weight.val(),
            eps: self.config.norm_eps,
            o_proj: &self.o_proj,
            hv,
            vd,
        }
    }
}

/// Per-head RMS norm with SiLU gate.
///
/// x: `[B, T, HV, V]`, gate: `[B, T, HV, V]`, weight: `[V]`
pub fn rms_norm_gate_per_head<B: Backend>(
    x: Tensor<B, 4>,
    gate: Tensor<B, 4>,
    weight: Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, 4> {
    let v = x.shape().dims::<4>()[3];
    let rms = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(3)
        .add_scalar(eps)
        .sqrt();
    let normed = x / rms;
    let w = weight.reshape([1, 1, 1, v]);
    normed * w * silu(gate)
}

pub struct ProjectedInputs<'a, B: Backend> {
    pub q: Tensor<B, 4>,
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
    pub g: Tensor<B, 4>,
    pub b: Tensor<B, 4>,
    pub w: Tensor<B, 4>,
    pub gate: Tensor<B, 4>,
    pub o_norm: Tensor<B, 1>,
    pub eps: f64,
    pub o_proj: &'a Linear<B>,
    pub hv: usize,
    pub vd: usize,
}
