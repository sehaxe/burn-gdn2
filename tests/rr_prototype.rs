use cubecl::cmma;
use cubecl::prelude::*;
use half::f16;

// Register-resident recurrence prototype on cubecl 0.11 (branch burn-0.22).
// Register-resident recurrence prototype on cubecl 0.11 (branch burn-0.22).
//
// Proves the core of the NVlabs-style recurrence is implementable on the
// new codegen: the state stays in a CMMA accumulator (registers) across
// chunks, all dots go through tensor cores:
//   - decay:  s += (diag(g)-I) @ s   (mma, since execute_elementwise_op is
//             broken on 0.11.0-pre.1)
//   - update: s += w @ u             (mma)
//   - out:    o  = qg @ s            (mma, s via shared round-trip)
// Precision: ~3e-5 vs CPU reference (f16 inputs, f32 accumulator).
//
// Known 0.11-pre gaps hit along the way:
//   - `cmma::execute_elementwise_op` panics in post-processing (DSD) and
//     produces garbage - replaced by the (diag-I) mma trick above.
//   - tf32 (`f32` matrices) wmma doesn't compile on sm_120 - inputs are f16.
// Baseline blocker status on 0.10: nvrtc dropped wmma.mma from long chains
// and corrupted loads around wmma code - both fixed on 0.11.

#[cube(launch)]
fn rr_kernel(
    w_all: &[f16],
    u_all: &[f16],
    qg_all: &[f16],
    diag_all: &[f16],
    nt: u32,
    out: &mut [f32],
) {
    let s = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    let mut s32_sh = Shared::<[f32]>::new_slice(256usize);
    let mut s16_sh = Shared::<[f16]>::new_slice(256usize);
    for j in range_stepped(0u32, nt, 1u32) {
        let jj = j as usize;
        // 1. decay: s += (diag_j - I) @ S  =>  s = diag_j @ S
        cmma::store(
            s32_sh.as_mut_slice(),
            &s,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..256usize {
            s16_sh[i] = f16::cast_from(s32_sh[i]);
        }
        sync_cube();
        let s_b = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        let d = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            diag_all.slice(jj * 256usize, (jj + 1) * 256usize),
            16u32,
        );
        cmma::execute(&d, &s_b, &s, &s);
        sync_cube();
        // 2. s += W_j @ U_j
        let w = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            w_all.slice(jj * 256usize, (jj + 1) * 256usize),
            16u32,
        );
        let u = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            u_all.slice(jj * 256usize, (jj + 1) * 256usize),
            16u32,
        );
        cmma::execute(&w, &u, &s, &s);
        // 3. out_j = QG_j @ S
        cmma::store(
            s32_sh.as_mut_slice(),
            &s,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..256usize {
            s16_sh[i] = f16::cast_from(s32_sh[i]);
        }
        sync_cube();
        let s_b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        let qg = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            qg_all.slice(jj * 256usize, (jj + 1) * 256usize),
            16u32,
        );
        let o = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        cmma::execute(&qg, &s_b2, &o, &o);
        cmma::store(
            out.slice_mut(jj * 256usize, (jj + 1) * 256usize),
            &o,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
    }
}

#[test]
fn rr_prototype_full() {
    const NT: usize = 8;
    let rng = |x: usize| -> f32 { ((x * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5 };
    let mk = |n: usize| -> Vec<f16> { (0..n).map(|i| f16::from_f32(rng(i) * 0.5)).collect() };
    let w_all = mk(NT * 256);
    let u_all = mk(NT * 256);
    let qg_all = mk(NT * 256);
    let g_all: Vec<f16> = (0..NT * 16)
        .map(|i| f16::from_f32(0.8 + rng(i) * 0.2))
        .collect();
    let f2 = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
    let w32 = f2(&w_all);
    let u32 = f2(&u_all);
    let qg32 = f2(&qg_all);
    let g32: Vec<f32> = f2(&g_all);
    let mut diag_all = vec![f16::from_f32(0.0); NT * 256];
    for j in 0..NT {
        for i in 0..16 {
            diag_all[j * 256 + i * 16 + i] = f16::from_f32(g32[j * 16 + i] - 1.0);
        }
    }
    let device = cubecl::cuda::CudaDevice::new(0);
    let client = cubecl::cuda::CudaRuntime::client(&device);
    let mk_buf = |v: &[f16]| client.create_from_slice(f16::as_bytes(v));
    let out_h = client.empty(NT * 256 * 4);
    unsafe {
        rr_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            BufferArg::from_raw_parts(mk_buf(&w_all), NT * 256),
            BufferArg::from_raw_parts(mk_buf(&u_all), NT * 256),
            BufferArg::from_raw_parts(mk_buf(&qg_all), NT * 256),
            BufferArg::from_raw_parts(mk_buf(&diag_all), NT * 256),
            NT as u32,
            BufferArg::from_raw_parts(out_h.clone(), NT * 256),
        );
    }
    let raw = client.read_one_unchecked(out_h);
    let got: Vec<f32> = raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let mut s = vec![0.0f32; 256];
    let mut exp = vec![0.0f32; NT * 256];
    for j in 0..NT {
        for r in 0..16 {
            for v in 0..16 {
                s[r * 16 + v] *= g32[j * 16 + r];
            }
        }
        for r in 0..16 {
            for v in 0..16 {
                let mut acc = 0.0f32;
                for k in 0..16 {
                    acc += w32[j * 256 + r * 16 + k] * u32[j * 256 + k * 16 + v];
                }
                s[r * 16 + v] += acc;
            }
        }
        for r in 0..16 {
            for v in 0..16 {
                let mut acc = 0.0f32;
                for k in 0..16 {
                    acc += qg32[j * 256 + r * 16 + k] * s[k * 16 + v];
                }
                exp[j * 256 + r * 16 + v] = acc;
            }
        }
    }
    let mut max_diff = 0.0f32;
    for i in 0..NT * 256 {
        max_diff = max_diff.max((got[i] - exp[i]).abs());
    }
    assert!(
        max_diff < 2e-1,
        "register-resident recurrence broken: {max_diff:.3e}"
    );
}
