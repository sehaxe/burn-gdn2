use burn::tensor::{backend::Backend, Tensor};

/// Cumulative sum along dim 2 via batched matmul with lower triangular matrix.
fn cumsum_seq_batched<B: Backend>(x: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [_b, _h, t, _d] = x.shape().dims::<4>();
    let device = x.device();
    let n = t * t;
    let mut d = vec![0.0f32; n];
    for i in 0..t { for j in 0..=i { d[i * t + j] = 1.0; } }
    let tril = Tensor::<B, 1>::from_floats(d.as_slice(), &device)
        .reshape([t, t])
        .unsqueeze_dims(&[0, 0]); // [1, 1, T, T]
    x.clone().swap_dims(2, 3).matmul(tril).swap_dims(2, 3)
}

/// Chunkwise WY forward — cumsum via batched matmul.
pub fn chunk_wy_forward<B: Backend>(
    q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>,
    g: Tensor<B, 4>, b: Tensor<B, 4>, w_gate: Tensor<B, 4>,
    mut state: Tensor<B, 4>, scale: f64, chunk_size: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, k_dim] = q.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];
    let device = q.device();
    let n_chunks = time.div_ceil(chunk_size);
    let mut outputs = Vec::with_capacity(n_chunks);

    for chunk_start in (0..time).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(time);
        let c = chunk_end - chunk_start;
        if c == 0 { continue; }

        let q_c = q.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let k_c = k.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let v_c = v.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let g_c = g.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let b_c = b.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let w_c = w_gate.clone().slice([0..batch, 0..heads, chunk_start..chunk_end]);

        let g_cumsum = if c > 1 { cumsum_seq_batched(&g_c) } else { g_c.clone() };

        let g_i = g_cumsum.clone().unsqueeze_dim::<5>(3);
        let g_j = g_cumsum.clone().unsqueeze_dim::<5>(2);
        let decay = (g_i - g_j).exp().mean_dim(4).reshape([batch, heads, c, c]);

        let n_m = c * c;
        let mut causal_v = vec![0.0f32; n_m];
        let mut strict_v = vec![0.0f32; n_m];
        for i in 0..c { for j in 0..=i { causal_v[i * c + j] = 1.0; if j < i { strict_v[i * c + j] = 1.0; } } }
        let causal_mask = Tensor::<B, 1>::from_floats(causal_v.as_slice(), &device).reshape([1, 1, c, c]);
        let strict_mask = Tensor::<B, 1>::from_floats(strict_v.as_slice(), &device).reshape([1, 1, c, c]);

        let qk = q_c.clone().matmul(k_c.clone().swap_dims(2, 3));
        let aqk = qk * decay.clone() * scale * causal_mask;

        let bk = b_c.clone() * k_c.clone();
        let bk_k = bk.matmul(k_c.clone().swap_dims(2, 3));
        let akk = bk_k * decay * strict_mask;

        let g_cumsum_for_last = g_cumsum.clone();
        let g_exp = g_cumsum.exp();

        let rhs_k = b_c * k_c.clone() * g_exp.clone();
        let rhs_v = w_c * v_c;

        // 7. Forward substitution — pre-allocate, use slice_assign
        let mut w_wy_full = Tensor::zeros([batch, heads, c, k_dim], &device);
        let mut u_full = Tensor::zeros([batch, heads, c, v_dim], &device);

        // Row 0
        let rk0 = rhs_k.clone().slice([0..batch, 0..heads, 0..1, 0..k_dim]);
        let rv0 = rhs_v.clone().slice([0..batch, 0..heads, 0..1, 0..v_dim]);
        w_wy_full = w_wy_full.slice_assign([0..batch, 0..heads, 0..1, 0..k_dim], rk0);
        u_full = u_full.slice_assign([0..batch, 0..heads, 0..1, 0..v_dim], rv0);

        for i in 1..c {
            let akk_row = akk.clone().slice([0..batch, 0..heads, i..i + 1, 0..i]);
            // w_prev: rows 0..i of w_wy_full [B, H, i, K]
            let w_prev = w_wy_full.clone().slice([0..batch, 0..heads, 0..i, 0..k_dim]);
            let w_new = rhs_k.clone().slice([0..batch, 0..heads, i..i + 1, 0..k_dim])
                - akk_row.clone().matmul(w_prev);
            w_wy_full = w_wy_full.slice_assign([0..batch, 0..heads, i..i + 1, 0..k_dim], w_new);

            let u_prev = u_full.clone().slice([0..batch, 0..heads, 0..i, 0..v_dim]);
            let u_new = rhs_v.clone().slice([0..batch, 0..heads, i..i + 1, 0..v_dim])
                - akk_row.matmul(u_prev);
            u_full = u_full.slice_assign([0..batch, 0..heads, i..i + 1, 0..v_dim], u_new);
        }
        let w_wy = w_wy_full;
        let u = u_full;

        let state_before = state.clone();
        let v_new = u - w_wy.matmul(state_before.clone());
        let intra = aqk.matmul(v_new.clone());
        let q_gated = q_c * g_exp.clone();
        let inter = q_gated.matmul(state_before) * scale;
        outputs.push(intra + inter);

        let g_last_exp = g_exp.clone().slice([0..batch, 0..heads, c - 1..c]);
        let g_cumsum_last = g_cumsum_for_last.clone().slice([0..batch, 0..heads, c - 1..c]);
        let decay_from_last = (g_cumsum_last - g_cumsum_for_last).exp();
        let k_eff = k_c * decay_from_last;
        state = state * g_last_exp.swap_dims(2, 3) + k_eff.swap_dims(2, 3).matmul(v_new);
    }

    (Tensor::cat(outputs, 2), state)
}

pub fn verify_chunk_vs_reference<B: Backend>(
    q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>,
    g: Tensor<B, 4>, b: Tensor<B, 4>, w_gate: Tensor<B, 4>,
    scale: f64, chunk_size: usize,
) -> f32 {
    let [batch, heads, _time, k_dim] = q.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];
    let device = q.device();
    let state = Tensor::zeros([batch, heads, k_dim, v_dim], &device);
    let (chunk_out, _) = chunk_wy_forward(
        q.clone(), k.clone(), v.clone(), g.clone(),
        b.clone(), w_gate.clone(), state.clone(), scale, chunk_size,
    );
    let (ref_out, _) = crate::kernel::fused_recurrent::fused_recurrent_forward(
        q, k, v, g, b, w_gate, state, scale,
    );
    (chunk_out - ref_out).abs().max().into_data().bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .fold(0.0f32, f32::max)
}
