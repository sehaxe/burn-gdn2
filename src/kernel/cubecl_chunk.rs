use cubecl::prelude::*;
use cubecl::server::Handle;

#[cube(launch_unchecked)]
fn chunk_substitution_kernel(
    akk: &Array<f32>,
    rhs_k: &Array<f32>,
    rhs_v: &Array<f32>,
    w_wy: &mut Array<f32>,
    u: &mut Array<f32>,
    n_batch: u32,
    n_heads_v: u32,
    n_chunks: u32,
    chunk_size: u32,
    dim_k: u32,
    dim_v: u32,
) {
    let chunk_id = CUBE_POS_X as usize;
    let bh_id = CUBE_POS_Y as usize;
    let nb = n_batch as usize;
    let nv = n_heads_v as usize;
    let nc = n_chunks as usize;
    let c = chunk_size as usize;
    let dk = dim_k as usize;
    let dv = dim_v as usize;
    let b = bh_id / nv;
    let hv_idx = bh_id % nv;

    if b >= nb || chunk_id >= nc || hv_idx >= nv {
        terminate!();
    }

    let cc = c * c;
    let ck = c * dk;
    let cv = c * dv;
    let hv_cc = nv * cc;
    let hv_ck = nv * ck;
    let hv_cv = nv * cv;
    let nc_hv_cc = nc * hv_cc;
    let nc_hv_ck = nc * hv_ck;
    let nc_hv_cv = nc * hv_cv;

    let akk_base = b * nc_hv_cc + chunk_id * hv_cc + hv_idx * cc;
    let rhs_k_base = b * nc_hv_ck + chunk_id * hv_ck + hv_idx * ck;
    let rhs_v_base = b * nc_hv_cv + chunk_id * hv_cv + hv_idx * cv;
    let wk_base = rhs_k_base;
    let uv_base = rhs_v_base;

    for k_idx in 0..dk {
        w_wy[wk_base + k_idx] = rhs_k[rhs_k_base + k_idx];
    }
    for v_idx in 0..dv {
        u[uv_base + v_idx] = rhs_v[rhs_v_base + v_idx];
    }

    for i in 1..c {
        let row_off = i * dk;
        let col_off = i * dv;

        for k_idx in 0..dk {
            w_wy[wk_base + row_off + k_idx] = rhs_k[rhs_k_base + row_off + k_idx];
        }
        for v_idx in 0..dv {
            u[uv_base + col_off + v_idx] = rhs_v[rhs_v_base + col_off + v_idx];
        }

        for j in 0..i {
            let akk_ij = akk[akk_base + i * c + j];
            let jr_off = j * dk;
            let jc_off = j * dv;
            for k_idx in 0..dk {
                w_wy[wk_base + row_off + k_idx] =
                    w_wy[wk_base + row_off + k_idx] - akk_ij * w_wy[wk_base + jr_off + k_idx];
            }
            for v_idx in 0..dv {
                u[uv_base + col_off + v_idx] =
                    u[uv_base + col_off + v_idx] - akk_ij * u[uv_base + jc_off + v_idx];
            }
        }
    }
}

pub unsafe fn launch_chunk_substitution<R: Runtime>(
    client: &ComputeClient<R>,
    akk_handle: &Handle,
    rhs_k_handle: &Handle,
    rhs_v_handle: &Handle,
    w_wy_handle: &Handle,
    u_handle: &Handle,
    n_batch: u32,
    n_heads_v: u32,
    n_chunks: u32,
    chunk_size: u32,
    dim_k: u32,
    dim_v: u32,
) {
    let cube_dim = CubeDim::new_1d(1);
    let cube_count = CubeCount::Static(n_chunks, n_batch * n_heads_v, 1);

    let total = n_batch * n_heads_v * n_chunks * chunk_size;
    let akk_len = (total * chunk_size) as usize;
    let rhs_k_len = (total * dim_k) as usize;
    let rhs_v_len = (total * dim_v) as usize;

    chunk_substitution_kernel::launch_unchecked::<R>(
        client,
        cube_count,
        cube_dim,
        ArrayArg::from_raw_parts(akk_handle.clone(), akk_len),
        ArrayArg::from_raw_parts(rhs_k_handle.clone(), rhs_k_len),
        ArrayArg::from_raw_parts(rhs_v_handle.clone(), rhs_v_len),
        ArrayArg::from_raw_parts(w_wy_handle.clone(), rhs_k_len),
        ArrayArg::from_raw_parts(u_handle.clone(), rhs_v_len),
        n_batch,
        n_heads_v,
        n_chunks,
        chunk_size,
        dim_k,
        dim_v,
    );
}
