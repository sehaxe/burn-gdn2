use cubecl::prelude::*;
use cubecl::server::Handle;

#[cube(launch_unchecked)]
fn fused_recurrent_kernel(
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    g: &Array<f32>,
    b: &Array<f32>,
    w: &Array<f32>,
    out: &mut Array<f32>,
    state: &mut Array<f32>,
    scale: f32,
    n_heads_q: u32,
    n_tokens: u32,
    dim_k: u32,
    dim_v: u32,
    hv: u32,
) {
    let h = CUBE_POS_X as usize;
    let b_idx = CUBE_POS_Y as usize;
    let v_idx = UNIT_POS_X as usize;
    let nq = n_heads_q as usize;
    let nt = n_tokens as usize;
    let dk = dim_k as usize;
    let dv = dim_v as usize;
    let n_v = hv as usize;

    if h >= nq || v_idx >= dv {
        terminate!();
    }

    let htk = nt * dk;
    let htv = nt * dv;
    let kv = dk * dv;

    let rep = n_v / nq;
    for h_rep in 0..rep {
        let hv_idx = h * rep + h_rep;
        if hv_idx >= n_v {
            break;
        }

        let batch_off_q = b_idx * nq * htk;
        let batch_off_v = b_idx * n_v * htv;
        let state_off = b_idx * n_v * kv + hv_idx * kv + v_idx;

        for t in 0..nt {
            let tk = t * dk;
            let tv = t * dv;

            let q_off = batch_off_q + h * htk + tk;
            let k_off = q_off;
            let g_off = q_off;
            let b_off = q_off;
            let v_off = batch_off_v + hv_idx * htv + tv;
            let w_off = v_off;

            let g_addr = g_off + v_idx % dk;
            let g_val = g[g_addr];
            let g_decay = f32::exp(g_val);

            let mut sum_erase = 0.0f32;
            let mut sum_out = 0.0f32;

            for k_idx in 0..dk {
                let s_off = state_off + k_idx * dv;
                let mut s_val = state[s_off];

                s_val = s_val * g_decay;
                state[s_off] = s_val;

                let bk_val = b[b_off + k_idx] * k[k_off + k_idx];
                sum_erase = sum_erase + s_val * bk_val;
                sum_out = sum_out + s_val * q[q_off + k_idx];
            }

            let v_new_val = w[w_off + v_idx] * v[v_off + v_idx] - sum_erase;

            for k_idx in 0..dk {
                let s_off = state_off + k_idx * dv;
                let k_val = k[k_off + k_idx];
                state[s_off] = state[s_off] + k_val * v_new_val;
            }

            out[batch_off_v + hv_idx * htv + tv + v_idx] = sum_out * scale;
        }
    }
}

pub unsafe fn launch_fused_recurrent<R: Runtime>(
    client: &ComputeClient<R>,
    q_handle: &Handle,
    k_handle: &Handle,
    v_handle: &Handle,
    g_handle: &Handle,
    b_handle: &Handle,
    w_handle: &Handle,
    out_handle: &Handle,
    state_handle: &Handle,
    scale: f32,
    n_batch: u32,
    n_heads_q: u32,
    n_tokens: u32,
    dim_k: u32,
    dim_v: u32,
    hv: u32,
) {
    let cube_dim = CubeDim::new_3d(dim_v, 1, 1);
    let cube_count = CubeCount::Static(n_heads_q, n_batch, 1);

    let bk = (n_batch * n_heads_q * n_tokens * dim_k) as usize;
    let bv = (n_batch * hv * n_tokens * dim_v) as usize;
    let sk = (n_batch * hv * dim_k * dim_v) as usize;

    fused_recurrent_kernel::launch_unchecked::<R>(
        client,
        cube_count,
        cube_dim,
        ArrayArg::from_raw_parts(q_handle.clone(), bk),
        ArrayArg::from_raw_parts(k_handle.clone(), bk),
        ArrayArg::from_raw_parts(v_handle.clone(), bv),
        ArrayArg::from_raw_parts(g_handle.clone(), bk),
        ArrayArg::from_raw_parts(b_handle.clone(), bk),
        ArrayArg::from_raw_parts(w_handle.clone(), bv),
        ArrayArg::from_raw_parts(out_handle.clone(), bv),
        ArrayArg::from_raw_parts(state_handle.clone(), sk),
        scale,
        n_heads_q,
        n_tokens,
        dim_k,
        dim_v,
        hv,
    );
}
