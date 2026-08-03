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

fn pad_time<B: Backend>(x: Tensor<B, 4>, c_pad: usize, device: &B::Device) -> Tensor<B, 4> {
    let [b, h, c, d] = x.shape().dims::<4>();
    if c == c_pad {
        x
    } else {
        Tensor::cat(vec![x, Tensor::zeros([b, h, c_pad - c, d], device)], 2)
    }
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

        // Tile-local log-space decay (K3 16-tile scheme). The naive
        // K/exp(cumsum(g)) factor overflows f32 once cumsum(g) < -88
        // (chunk > 17 at g_min = -5): exp(320) = inf. Here every decay
        // factor is exp(x) with x <= 0, so the chunk length is unbounded.
        let tile = 16usize;
        let c_pad = c.div_ceil(tile) * tile;
        let n_t = c_pad / tile;

        let g_p = pad_time(g_c, c_pad, &device);
        let q_p = pad_time(q_c, c_pad, &device);
        let k_p = pad_time(k_c, c_pad, &device);
        let v_p = pad_time(v_c, c_pad, &device);
        let b_p = pad_time(b_c, c_pad, &device);
        let w_p = pad_time(w_c, c_pad, &device);

        // full cumulative log-decay over the whole (padded) chunk
        let (causal_mask, strict_mask) = chunk_masks(c_pad, &device);
        let tril_full_c = tril_matrix(c_pad, &device);
        let g_cumsum = g_p
            .clone()
            .swap_dims(2, 3)
            .matmul(tril_full_c.transpose())
            .swap_dims(2, 3); // [B,H,c_pad,k_dim]
        // exclusive prefix of tile sums: decay accumulated before each tile,
        // i.e. the full cumsum at the last position of the previous tile
        let g_dim = g_cumsum.shape().dims::<4>()[3];
        let g_bound_prev = g_cumsum
            .clone()
            .reshape([batch, heads, n_t, tile, g_dim])
            .slice([
                0..batch,
                0..heads,
                0..n_t - 1,
                tile - 1..tile,
                0..g_dim,
            ])
            .reshape([batch, heads, n_t - 1, g_dim]);
        let g_bound = Tensor::cat(
            vec![
                Tensor::zeros([batch, heads, 1, g_dim], &device),
                g_bound_prev,
            ],
            2,
        ); // [B,H,n_t,g_dim]

        // tile-local decay: full cumsum minus the boundary accumulated before
        // the tile (never overflows: at most 16 * g_min = -80 -> exp <= e^-80)
        let g_bound_full = g_bound
            .clone()
            .reshape([batch, heads, n_t, 1, g_dim])
            .repeat(&[1, 1, 1, tile, 1])
            .reshape([batch, heads, c_pad, g_dim]);
        let g_rel_log_full = g_cumsum.clone() - g_bound_full.clone();
        let g_rel_exp_full = g_rel_log_full.clone().exp();

        let g_full_log = g_cumsum;
        let gamma = g_full_log.clone().exp(); // full decay, <= 1

        // inter-tile decay exp(G_p - G_q) per channel, clamped to 1 above the
        // diagonal so upper blocks are killed by the causal mask (never inf*0)
        let e_block_full = |k: usize| {
            let gb_k = g_bound
                .clone()
                .slice([0..batch, 0..heads, 0..n_t, (k % g_dim)..(k % g_dim) + 1]);
            (gb_k.clone() - gb_k.clone().swap_dims(2, 3))
                .clamp_max(0.0)
                .exp()
                .reshape([batch, heads, n_t, 1, n_t, 1])
                .repeat(&[1, 1, 1, tile, 1, 1])
                .repeat(&[1, 1, 1, 1, 1, tile])
                .reshape([batch, heads, c_pad, c_pad])
        };

        // A_qk and A_kk are channel sums with a per-channel inter-tile weight,
        // so they cannot be one matmul: accumulate over the decay channels.
        let mut aqk = Tensor::zeros([batch, heads, c_pad, c_pad], &device);
        let mut akk = Tensor::zeros([batch, heads, c_pad, c_pad], &device);
        for k in 0..k_dim {
            let q_k = q_p
                .clone()
                .slice([0..batch, 0..heads, 0..c_pad, k..k + 1]);
            let k_k = k_p
                .clone()
                .slice([0..batch, 0..heads, 0..c_pad, k..k + 1]);
            let g_dim = g_rel_exp_full.shape().dims::<4>()[3];
            let g_k = g_rel_exp_full
                .clone()
                .slice([0..batch, 0..heads, 0..c_pad, (k % g_dim)..(k % g_dim) + 1]);
            let e_k = e_block_full(k);
            let gk_a = g_k.clone();
            let gk_b = g_k.clone();
            aqk = aqk
                + (q_k * gk_a.clone()).matmul((k_k.clone() / gk_a).swap_dims(2, 3)) * e_k.clone();
            let b_dim = b_p.shape().dims::<4>()[3];
            let b_k = b_p
                .clone()
                .slice([0..batch, 0..heads, 0..c_pad, (k % b_dim)..(k % b_dim) + 1]);
            let bkk = (b_k * k_k.clone()) * gk_b.clone();
            akk = akk + bkk.matmul((k_k.clone() / gk_b).swap_dims(2, 3)) * e_k;
        }
        let aqk = aqk * scale * causal_mask;
        let akk = akk * strict_mask;

        let rhs_k = b_p * k_p.clone() * gamma.clone();
        let rhs_v = w_p * v_p;

        // Blocked WY solve (K3 16-tile scheme): invert each 16x16 diagonal
        // block locally and fold in the strictly-lower inter-block
        // contributions tile by tile. A full-chunk solve blows up f32 error:
        // (I - T)^-1 has |T| ~ |k| sum|k_j| > 1 (T has rank <= K, powers up to
        // T^K, so error grows ~|T|^K — 6^8 on a 64-chunk vs 1.6^8 on a tile).
        let mut w_wy = Tensor::zeros([batch, heads, c_pad, k_dim], &device);
        let mut u = Tensor::zeros([batch, heads, c_pad, v_dim], &device);

        for p in 0..n_t {
            let row0 = p * tile;
            let t_pp = akk
                .clone()
                .slice([0..batch, 0..heads, row0..row0 + tile, row0..row0 + tile]);
            let t_pq = akk
                .clone()
                .slice([0..batch, 0..heads, row0..row0 + tile, 0..row0]);
            let w_prev = w_wy.clone().slice([0..batch, 0..heads, 0..row0, 0..k_dim]);
            let u_prev = u.clone().slice([0..batch, 0..heads, 0..row0, 0..v_dim]);
            let mut wb = rhs_k
                .clone()
                .slice([0..batch, 0..heads, row0..row0 + tile, 0..k_dim])
                - if row0 > 0 {
                    t_pq.clone().matmul(w_prev)
                } else {
                    Tensor::zeros([batch, heads, tile, k_dim], &device)
                };
            let mut ub = rhs_v
                .clone()
                .slice([0..batch, 0..heads, row0..row0 + tile, 0..v_dim])
                - if row0 > 0 {
                    t_pq.matmul(u_prev)
                } else {
                    Tensor::zeros([batch, heads, tile, v_dim], &device)
                };

            for i in 0..tile {
                let t_row = t_pp.clone().slice([0..batch, 0..heads, i..i + 1, 0..i]);
                let wb_prev = wb.clone().slice([0..batch, 0..heads, 0..i, 0..k_dim]);
                let ub_prev = ub.clone().slice([0..batch, 0..heads, 0..i, 0..v_dim]);
                let w_row = wb
                    .clone()
                    .slice([0..batch, 0..heads, i..i + 1, 0..k_dim]);
                let u_row = ub
                    .clone()
                    .slice([0..batch, 0..heads, i..i + 1, 0..v_dim]);

                wb = wb.slice_assign(
                    [0..batch, 0..heads, i..i + 1, 0..k_dim],
                    if i == 0 {
                        w_row
                    } else {
                        w_row - t_row.clone().matmul(wb_prev)
                    },
                );
                ub = ub.slice_assign(
                    [0..batch, 0..heads, i..i + 1, 0..v_dim],
                    if i == 0 {
                        u_row
                    } else {
                        u_row - t_row.matmul(ub_prev)
                    },
                );
            }

            w_wy = w_wy.slice_assign(
                [0..batch, 0..heads, row0..row0 + tile, 0..k_dim],
                wb,
            );
            u = u.slice_assign([0..batch, 0..heads, row0..row0 + tile, 0..v_dim], ub);
        }

        let state_before = state.clone();
        let v_new = u.clone() - w_wy.clone().matmul(state_before.clone());
        let intra = aqk.matmul(v_new.clone());
        let q_gated = q_p * gamma.clone();
        let inter = q_gated.matmul(state_before) * scale;
        let out_c = (intra + inter).slice([0..batch, 0..heads, 0..c, 0..v_dim]);
        outputs.push(out_c);

        let g_last = gamma.clone().slice([0..batch, 0..heads, c - 1..c]);
        let g_last_log = g_full_log.clone().slice([0..batch, 0..heads, c - 1..c]);
        let decay_last = (g_last_log - g_full_log).exp();
        state = state * g_last.swap_dims(2, 3) + (k_p * decay_last).swap_dims(2, 3).matmul(v_new);
    }

    (Tensor::cat(outputs, 2), state)
}
