use burn::tensor::{Device, Tensor};

fn tril_matrix(t: usize, device: &Device) -> Tensor<4> {
    let n = t * t;
    let mut d = vec![0.0f32; n];
    for i in 0..t {
        for j in 0..=i {
            d[i * t + j] = 1.0;
        }
    }
    Tensor::<1>::from_floats(d.as_slice(), device)
        .reshape([t, t])
        .unsqueeze_dims(&[0, 0])
}

pub(crate) fn chunk_masks(c: usize, device: &Device) -> (Tensor<4>, Tensor<4>) {
    let n = c * c;
    let mut causal = vec![0.0f32; n];
    let mut strict = vec![0.0f32; n];
    for i in 0..c {
        for j in 0..=i {
            causal[i * c + j] = 1.0;
            if j < i {
                strict[i * c + j] = 1.0;
            }
        }
    }
    (
        Tensor::<1>::from_floats(causal.as_slice(), device).reshape([1, 1, c, c]),
        Tensor::<1>::from_floats(strict.as_slice(), device).reshape([1, 1, c, c]),
    )
}

/// Intermediates of one chunk needed by the exact backward adjoint.
///
/// The adjoint re-derives gradients from these values without re-differentiating
/// the token loop: the WY solve is differentiated via the triangular adjoint
/// `d_rhs = M^-T d_*`, `d_akk = tril_strict(-M^-T d_* W^T)`.
#[derive(Clone, Debug)]
pub struct ChunkScratch {
    /// exp(cumsum(g)) over the chunk, `[B, H, c, k]`.
    pub g_exp: Tensor<4>,
    /// q * g_exp, `[B, H, c, k]`.
    pub q_gated: Tensor<4>,
    /// `(q·E)(k/E)^T * scale * causal`, `[B, H, c, c]`.
    pub aqk: Tensor<4>,
    /// `M^-1` with `M = I + L`: W = M⁻¹·rhs_k, backward solves via M⁻ᵀ·d_*.
    pub m_inv: Tensor<4>,
}

#[derive(Clone, Debug)]
pub struct ChunkWyScratch {
    pub chunks: Vec<ChunkScratch>,
}

#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_impl(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w_gate: Tensor<4>,
    mut state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
    m_invs: Option<&[Tensor<4>]>,
) -> (Tensor<4>, Tensor<4>, ChunkWyScratch) {
    let [batch, heads, time, _k_dim] = q.shape().dims::<4>();
    let device = q.device();
    let mut outputs = Vec::with_capacity(time.div_ceil(chunk_size));
    // The module feeds permuted [B,H,T,K] views here; cubecl ops would copy
    // them per-op. Materialize once so every later op sees contiguous buffers.
    let q = q.mul_scalar(1.0);
    let k = k.mul_scalar(1.0);
    let v = v.mul_scalar(1.0);
    let g = g.mul_scalar(1.0);
    let b = b.mul_scalar(1.0);
    let w_gate = w_gate.mul_scalar(1.0);
    let mut scratch = ChunkWyScratch { chunks: Vec::new() };
    let tril_full = tril_matrix(chunk_size, &device);
    let masks_full = chunk_masks(chunk_size, &device);
    // scale folded into the causal mask, eye pre-broadcast: hoisted so the
    // chunk loop pays one mul / one eye instead of two / three ops.
    let scale_causal_full = tril_full.clone() * scale;
    let eye_full = Tensor::<2>::eye(chunk_size, &device)
        .reshape([1, 1, chunk_size, chunk_size])
        .repeat(&[batch, heads, 1, 1]);

    for chunk_start in (0..time).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(time);
        let c = chunk_end - chunk_start;
        if c == 0 {
            continue;
        }

        let q_c = q
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let k_c = k
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let v_c = v
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let g_c = g
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let b_c = b
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let w_c = w_gate
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);

        let (scale_causal, eye) = if c == chunk_size {
            (scale_causal_full.clone(), eye_full.clone())
        } else {
            let tril_c = tril_matrix(c, &device);
            (
                tril_c.clone() * scale,
                Tensor::<2>::eye(c, &device)
                    .reshape([1, 1, c, c])
                    .repeat(&[batch, heads, 1, 1]),
            )
        };
        let strict_mask = if c == chunk_size {
            masks_full.1.clone()
        } else {
            chunk_masks(c, &device).1
        };

        let g_cumsum = g_c.clone().cumsum(2);
        let g_exp = g_cumsum.clone().exp();
        let k_over_gamma = k_c.clone() / g_exp.clone();
        let qk = (q_c.clone() * g_exp.clone()).matmul(k_over_gamma.clone().swap_dims(2, 3));
        let aqk = qk * scale_causal;

        let bk = b_c.clone() * k_c.clone();
        let akk = (bk.clone() * g_exp.clone()).matmul(k_over_gamma.clone().swap_dims(2, 3))
            * strict_mask;

        let rhs_k = bk.clone() * g_exp.clone();
        let rhs_v = w_c * v_c;

        // M = I + L (unit lower triangular, L = strict-lower(akk)). Invert M
        // once per chunk (row i needs only rows < i), then all four solves —
        // forward W = M⁻¹·rhs_k, U = M⁻¹·rhs_v and the two backward solves
        // d_rhs = M⁻ᵀ·d_* — become single matmuls instead of per-row loops.
        // m_inv is seeded with the identity so the diagonal e_i survives the
        // strict-lower slice_assign.
        let mut m_inv = eye;
        if let Some(saved) = m_invs {
            // M^-1 exported by the fused kernel (or the previous forward):
            // skip the row-by-row inversion entirely.
            m_inv = saved[scratch.chunks.len()].clone();
        } else {
            for i in 1..c {
                let akk_row = akk.clone().slice([0..batch, 0..heads, i..i + 1, 0..i]);
                let m_prev = m_inv.clone().slice([0..batch, 0..heads, 0..i, 0..c]);
                let row = -(akk_row.matmul(m_prev)).slice([0..batch, 0..heads, 0..1, 0..i]);
                m_inv = m_inv.slice_assign([0..batch, 0..heads, i..i + 1, 0..i], row);
            }
        }

        let w_wy = m_inv.clone().matmul(rhs_k.clone());
        let u = m_inv.clone().matmul(rhs_v.clone());

        let state_before = state.clone();
        let v_new = u.clone() - w_wy.clone().matmul(state_before.clone());
        let intra = aqk.clone().matmul(v_new.clone());
        let q_gated = q_c * g_exp.clone();
        let inter = q_gated.clone().matmul(state_before.clone()) * scale;
        outputs.push(intra + inter);

        // Only the four values the backward cannot cheaply re-derive are
        // kept (W/U/v_new/rhs/kG/akk all recompute from these + the input
        // checkpoints in ~8 ops per chunk); the fused kernels export the same
        // four, so the training path never re-runs the forward.
        scratch.chunks.push(ChunkScratch {
            g_exp: g_exp.clone(),
            q_gated,
            aqk,
            m_inv,
        });

        let g_last = g_exp.clone().slice([0..batch, 0..heads, c - 1..c]);
        let g_last_cumsum = g_cumsum.clone().slice([0..batch, 0..heads, c - 1..c]);
        let decay_last = (g_last_cumsum - g_cumsum).exp();
        state = state * g_last.swap_dims(2, 3) + (k_c * decay_last).swap_dims(2, 3).matmul(v_new);
    }

    (Tensor::cat(outputs, 2), state, ChunkWyScratch { chunks: scratch.chunks })
}

#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w_gate: Tensor<4>,
    state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
) -> (Tensor<4>, Tensor<4>) {
    let (output, new_state, _scratch) =
        chunk_wy_forward_impl(q, k, v, g, b, w_gate, state, scale, chunk_size, None);
    (output, new_state)
}
