use burn::tensor::{backend::Backend, Tensor};

fn tril_matrix<B: Backend>(t: usize, device: &B::Device) -> Tensor<B, 4> {
    let n = t * t;
    let mut d = vec![0.0f32; n];
    for i in 0..t {
        for j in 0..=i {
            d[i * t + j] = 1.0;
        }
    }
    Tensor::<B, 1>::from_floats(d.as_slice(), device)
        .reshape([t, t])
        .unsqueeze_dims(&[0, 0])
}

fn chunk_masks<B: Backend>(c: usize, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
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
        Tensor::<B, 1>::from_floats(causal.as_slice(), device).reshape([1, 1, c, c]),
        Tensor::<B, 1>::from_floats(strict.as_slice(), device).reshape([1, 1, c, c]),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    g: Tensor<B, 4>,
    b: Tensor<B, 4>,
    w_gate: Tensor<B, 4>,
    mut state: Tensor<B, 4>,
    scale: f64,
    chunk_size: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, k_dim] = q.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];
    let device = q.device();
    let mut outputs = Vec::with_capacity(time.div_ceil(chunk_size));
    let tril_full = tril_matrix(chunk_size, &device);
    let masks_full = chunk_masks(chunk_size, &device);

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

        let (tril, (causal_mask, strict_mask)) = if c == chunk_size {
            (
                tril_full.clone(),
                (masks_full.0.clone(), masks_full.1.clone()),
            )
        } else {
            (tril_matrix(c, &device), chunk_masks(c, &device))
        };

        let g_cumsum = g_c.clone().swap_dims(2, 3).matmul(tril).swap_dims(2, 3);
        let gi = g_cumsum.clone().unsqueeze_dim::<5>(3);
        let gj = g_cumsum.clone().unsqueeze_dim::<5>(2);
        let decay = (gi - gj).exp().mean_dim(4).reshape([batch, heads, c, c]);

        // Pre-compute k_c transposed — used twice
        let k_c_t = k_c.clone().swap_dims(2, 3);
        let qk = q_c.clone().matmul(k_c_t.clone());
        let aqk = qk * decay.clone() * scale * causal_mask;

        let bk = b_c.clone() * k_c.clone();
        let akk = bk.matmul(k_c_t) * decay * strict_mask;

        let g_cumsum_for_decay = g_cumsum.clone();
        let g_exp = g_cumsum.exp();
        let rhs_k = b_c * k_c.clone() * g_exp.clone();
        let rhs_v = w_c * v_c;

        let mut w_wy = Tensor::zeros([batch, heads, c, k_dim], &device);
        let mut u = Tensor::zeros([batch, heads, c, v_dim], &device);

        for i in 0..c {
            let akk_row = akk.clone().slice([0..batch, 0..heads, i..i + 1, 0..i]);
            let w_prev = w_wy.clone().slice([0..batch, 0..heads, 0..i, 0..k_dim]);
            let u_prev = u.clone().slice([0..batch, 0..heads, 0..i, 0..v_dim]);
            let w_row = rhs_k
                .clone()
                .slice([0..batch, 0..heads, i..i + 1, 0..k_dim]);
            let u_row = rhs_v
                .clone()
                .slice([0..batch, 0..heads, i..i + 1, 0..v_dim]);

            w_wy = w_wy.slice_assign(
                [0..batch, 0..heads, i..i + 1, 0..k_dim],
                if i == 0 {
                    w_row
                } else {
                    w_row - akk_row.clone().matmul(w_prev)
                },
            );
            u = u.slice_assign(
                [0..batch, 0..heads, i..i + 1, 0..v_dim],
                if i == 0 {
                    u_row
                } else {
                    u_row - akk_row.matmul(u_prev)
                },
            );
        }

        let state_before = state.clone();
        let v_new = u.clone() - w_wy.clone().matmul(state_before.clone());
        let intra = aqk.matmul(v_new.clone());
        let q_gated = q_c * g_exp.clone();
        let inter = q_gated.matmul(state_before) * scale;
        outputs.push(intra + inter);

        let g_last = g_exp.clone().slice([0..batch, 0..heads, c - 1..c]);
        let g_last_cumsum = g_cumsum_for_decay
            .clone()
            .slice([0..batch, 0..heads, c - 1..c]);
        let decay_last = (g_last_cumsum - g_cumsum_for_decay).exp();
        state = state * g_last.swap_dims(2, 3) + (k_c * decay_last).swap_dims(2, 3).matmul(v_new);
    }

    (Tensor::cat(outputs, 2), state)
}

#[allow(clippy::too_many_arguments)]
pub fn verify_chunk_vs_reference<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    g: Tensor<B, 4>,
    b: Tensor<B, 4>,
    w_gate: Tensor<B, 4>,
    scale: f64,
    chunk_size: usize,
) -> f32 {
    let [batch, heads, _time, k_dim] = q.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];
    let device = q.device();
    let state = Tensor::zeros([batch, heads, k_dim, v_dim], &device);
    let (chunk_out, _) = chunk_wy_forward(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b.clone(),
        w_gate.clone(),
        state.clone(),
        scale,
        chunk_size,
    );
    let (ref_out, _) = crate::kernel::fused_recurrent::fused_recurrent_forward(
        q, k, v, g, b, w_gate, state, scale,
    );
    (chunk_out - ref_out)
        .abs()
        .max()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .fold(0.0f32, f32::max)
}
