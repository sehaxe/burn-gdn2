use crate::forward::build_masks;
use crate::kernel::{cubecl_dispatch, fused_recurrent};

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use burn_backend::TensorPrimitive;
use burn_backend::DType;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;
use burn_std::Shape;

/// Chunk WY forward using CubeCL for the GPU-accelerated substitution step.
///
/// Mirrors [`crate::forward::chunk_wy_forward`] but replaces the inner
/// forward‑substitution loop with a single CubeCL kernel launch.
/// Pre‑/post‑substitution matmuls remain standard Burn ops.
#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_cubecl<B, R>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    g: Tensor<B, 4>,
    b_erase: Tensor<B, 4>,
    w_gate: Tensor<B, 4>,
    mut state: Tensor<B, 4>,
    scale: f64,
    chunk_size: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>)
where
    B: Backend<FloatTensorPrimitive = CubeTensor<R>>,
    R: CubeRuntime + 'static,
{
    let [batch, heads, time, k_dim] = q.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];
    let device = q.device();
    let mut outputs = Vec::with_capacity(time);

    let max_chunk_size = chunk_size.min(time);
    let (causal_mask_tmpl, strict_mask_tmpl) = build_masks::<B>(max_chunk_size, &device);

    for chunk_start in (0..time).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(time);
        let c = chunk_end - chunk_start;
        if c == 0 {
            continue;
        }

        let q_c = q.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let k_c = k.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let v_c = v.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let g_c = g.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let b_c = b_erase.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let w_c = w_gate.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);

        // Cumulative sum of g within chunk [B, H, C, K]
        let g_cumsum = cumsum_seq_inner(&g_c);

        // Decay[i,j] = mean over K of exp(g_cumsum[i] - g_cumsum[j])
        let g_i = g_cumsum.clone().unsqueeze_dim::<5>(3);
        let g_j = g_cumsum.clone().unsqueeze_dim::<5>(2);
        let decay = (g_i - g_j).exp().mean_dim(4);
        let decay = decay.reshape([batch, heads, c, c]);

        // Slice masks to actual chunk size
        let causal_mask = causal_mask_tmpl.clone().slice([0..1, 0..1, 0..c, 0..c]);
        let strict_mask = strict_mask_tmpl.clone().slice([0..1, 0..1, 0..c, 0..c]);

        // Aqk[i,j] = q[i]·k[j] · decay[i,j] · 1/√K
        let qk = q_c.clone().matmul(k_c.clone().swap_dims(2, 3));
        let aqk = qk * decay.clone() * scale * causal_mask;

        // Akk[i,j] = (b[i]·k[i])·k[j] · decay[i,j]  (j < i)
        let bk = b_c.clone() * k_c.clone();
        let bk_k = bk.matmul(k_c.clone().swap_dims(2, 3));
        let akk = bk_k * decay * strict_mask;

        // RHS vectors
        let g_exp = g_cumsum.clone().exp();
        let rhs_k = b_c.clone() * k_c.clone() * g_exp.clone();
        let rhs_v = w_c.clone() * v_c.clone();

        let akk_p: CubeTensor<R> = akk.into_primitive().tensor();
        let rhs_k_p: CubeTensor<R> = rhs_k.into_primitive().tensor();
        let rhs_v_p: CubeTensor<R> = rhs_v.into_primitive().tensor();
        let client = akk_p.client.clone();
        let dev = akk_p.device.clone();

        let wk_elems = batch * heads * c * k_dim;
        let uv_elems = batch * heads * c * v_dim;

        let (w_handle, u_handle) = unsafe {
            let wh = client.empty(wk_elems * 4);
            let uh = client.empty(uv_elems * 4);
            cubecl_dispatch::run_chunk_substitution::<R>(
                &client,
                batch as u32,
                heads as u32,
                1u32,
                c as u32,
                k_dim as u32,
                v_dim as u32,
                &akk_p.handle,
                &rhs_k_p.handle,
                &rhs_v_p.handle,
                &wh,
                &uh,
            );
            (wh, uh)
        };

        let sk = Shape::from(vec![batch, heads, c, k_dim]);
        let sv = Shape::from(vec![batch, heads, c, v_dim]);

        let w_t: CubeTensor<R> = CubeTensor::new_contiguous(
            client.clone(),
            dev.clone(),
            sk,
            w_handle,
            DType::F32,
        );
        let u_t: CubeTensor<R> = CubeTensor::new_contiguous(
            client, dev, sv, u_handle, DType::F32,
        );

        let w_wy: Tensor<B, 4> = Tensor::from_primitive(TensorPrimitive::Float(w_t));
        let u: Tensor<B, 4> = Tensor::from_primitive(TensorPrimitive::Float(u_t));

        // Intra-chunk output: v_new = u - w_wy × state_before
        let state_before = state.clone();
        let v_new_t = u.clone() - w_wy.matmul(state_before.clone());
        let intra = aqk.matmul(v_new_t);
        let q_gated = q_c * g_exp;
        let inter = q_gated.matmul(state_before) * scale;
        outputs.push(intra + inter);

        // Inter-chunk state update
        let g_last = g_cumsum.clone().slice([0..batch, 0..heads, c - 1..c]);
        let k_eff = k_c * (g_last.clone() - g_cumsum).exp();
        state = state * g_last.exp().swap_dims(2, 3) + k_eff.swap_dims(2, 3).matmul(u);
    }

    (Tensor::cat(outputs, 2), state)
}

/// Training forward using CubeCL-accelerated Chunk substitution.
///
/// Call this instead of [`GatedDeltaNet2::forward_train`] when using
/// a CubeCL backend (e.g. `CubeBackend<CudaRuntime>`).
///
/// Falls back to reference fused recurrent for sequences shorter than
/// the chunk size.
#[allow(clippy::too_many_arguments)]
pub fn forward_train_cubecl<B, R>(
    model: &crate::module::GatedDeltaNet2<B>,
    hidden_states: Tensor<B, 3>,
) -> Tensor<B, 3>
where
    B: Backend<FloatTensorPrimitive = CubeTensor<R>>,
    R: CubeRuntime + 'static,
{
    let [batch, tokens, _] = hidden_states.shape().dims::<3>();
    let projected = model.project(hidden_states);
    let hk = model.config.head_dim;
    let hv = projected.hv;
    let vd = projected.vd;
    let scale = (hk as f64).powf(-0.5);
    let device = projected.q.device();
    let state = Tensor::zeros([batch, hv, hk, vd / hv], &device);

    let output = match model.config.mode {
        crate::config::Gdn2Mode::FusedRecurrent => {
            let (o, _s) = fused_recurrent::fused_recurrent_gdn2(
                projected.q, projected.k, projected.v, projected.g,
                projected.b, projected.w, Some(state), scale, true,
            );
            o
        }
        crate::config::Gdn2Mode::Chunk => {
            let (o, _s) = chunk_wy_forward_cubecl::<B, R>(
                projected.q, projected.k, projected.v, projected.g,
                projected.b, projected.w, state, scale, model.config.chunk_size,
            );
            o
        }
    };

    let out_4d = output.permute([0, 2, 1, 3]);
    let out_norm = crate::module::rms_norm_gate_per_head(
        out_4d, projected.gate, projected.o_norm, projected.eps,
    );
    projected.o_proj.forward(out_norm.reshape([batch, tokens, vd]))
}

fn cumsum_seq_inner<B: Backend>(x: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, t, d] = x.shape().dims::<4>();
    let device = x.device();
    let mut slices = Vec::with_capacity(t);
    let mut acc = Tensor::zeros([b, h, 1, d], &device);
    for i in 0..t {
        acc = acc + x.clone().slice([0..b, 0..h, i..i + 1, 0..d]);
        slices.push(acc.clone());
    }
    Tensor::cat(slices, 2)
}


