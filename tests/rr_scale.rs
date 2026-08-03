//! Scaled register-resident recurrence prototype: kd=64, vd=16, c=64.
//! One warp per head; state = 4 CMMA accumulators (16x16x16 each).
//! All dots via tensor cores, k-loop over 16-wide steps.

use cubecl::cmma;
use cubecl::prelude::*;
use half::f16;

const KD: usize = 64;
const VD: usize = 16;
const C: usize = 64;

#[cube(launch)]
fn rr64_kernel(
    w_all: &[f32],    // [heads*nt, C, KD]
    u_all: &[f32],    // [heads*nt, C, VD]
    qg_all: &[f32],   // [heads*nt, C, KD]
    g_all: &[f32],    // [heads*nt, KD]
    state_in: &[f32], // [heads, KD, VD]
    state_out: &mut [f32],
    out: &mut [f32], // [heads*nt, C, VD]
    nt: u32,
) {
    let head = CUBE_POS_X as usize;
    let w_stride = C * KD;
    let u_stride = C * VD;
    let qg_stride = C * KD;
    let out_stride = C * VD;
    let base = head;

    let mut w_sh = Shared::<[f16]>::new_slice(C * KD);
    let mut u_sh = Shared::<[f16]>::new_slice(C * VD);
    let mut qg_sh = Shared::<[f16]>::new_slice(C * KD);
    let mut s32_sh = Shared::<[f32]>::new_slice(KD * VD);
    let mut s16_sh = Shared::<[f16]>::new_slice(KD * VD);
    let mut diag_sh = Shared::<[f16]>::new_slice(KD * KD);
    

    let s0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    let s1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    let s2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    let s3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    // init: s_m = S_in[m*16..]
    let mut eye16_sh = Shared::<[f16]>::new_slice(256usize);
    for i in 0..256usize {
        eye16_sh[i] = f16::cast_from(0.0f32);
    }
    for i in 0..16usize {
        eye16_sh[i * 16 + i] = f16::cast_from(1.0f32);
    }
    for i in 0..KD * VD {
        s32_sh[i] = state_in[base * KD * VD + i];
    }
    sync_cube();
    for i in 0..KD * VD {
        s16_sh[i] = f16::cast_from(s32_sh[i]);
    }
    sync_cube();
    let eye = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &eye16_sh[0..256],
        16u32,
    );
    let b0 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[0..256],
        16u32,
    );
    let b1 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[256..512],
        16u32,
    );
    let b2 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[512..768],
        16u32,
    );
    let b3 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[768..1024],
        16u32,
    );
    cmma::execute(&eye, &b0, &s0, &s0);
    cmma::execute(&eye, &b1, &s1, &s1);
    cmma::execute(&eye, &b2, &s2, &s2);
    cmma::execute(&eye, &b3, &s3, &s3);

    for j in range_stepped(0u32, nt, 1u32) {
        let jj = j as usize;
        let hw = base * (nt as usize) + jj;
        for i in 0..C * KD {
            let t = i / KD;
            let kd = i % KD;
            w_sh[kd * C + t] = f16::cast_from(w_all[hw * w_stride + i]);
        }
        for i in 0..C * VD {
            u_sh[i] = f16::cast_from(u_all[hw * u_stride + i]);
        }
        for i in 0..C * KD {
            qg_sh[i] = f16::cast_from(qg_all[hw * qg_stride + i]);
        }
        for i in 0..KD * KD {
            diag_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..KD {
            diag_sh[i * KD + i] = f16::cast_from(g_all[hw * KD + i]) - f16::cast_from(1.0f32);
        }
        sync_cube();
        // S -> shared (f16)
        cmma::store(
            s32_sh.as_mut_slice(),
            &s0,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(256usize, 512usize),
            &s1,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(512usize, 768usize),
            &s2,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(768usize, 1024usize),
            &s3,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..KD * VD {
            s16_sh[i] = f16::cast_from(s32_sh[i]);
        }
        sync_cube();
        // decay: s_{kk} += (diag16_kk - I) @ S_{kk}
        let mut diag16_sh0 = Shared::<[f16]>::new_slice(256usize);
        for i in 0..256usize {
            diag16_sh0[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh0[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + i]) - f16::cast_from(1.0f32);
        }
        let d0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh0[0..256],
            16u32,
        );
        let b0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&d0, &b0, &s0, &s0);
        let mut diag16_sh1 = Shared::<[f16]>::new_slice(256usize);
        for i in 0..256usize {
            diag16_sh1[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh1[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 16 + i]) - f16::cast_from(1.0f32);
        }
        let d1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh1[0..256],
            16u32,
        );
        let b1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&d1, &b1, &s1, &s1);
        let mut diag16_sh2 = Shared::<[f16]>::new_slice(256usize);
        for i in 0..256usize {
            diag16_sh2[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh2[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 32 + i]) - f16::cast_from(1.0f32);
        }
        let d2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh2[0..256],
            16u32,
        );
        let b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&d2, &b2, &s2, &s2);
        let mut diag16_sh3 = Shared::<[f16]>::new_slice(256usize);
        for i in 0..256usize {
            diag16_sh3[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh3[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 48 + i]) - f16::cast_from(1.0f32);
        }
        let d3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh3[0..256],
            16u32,
        );
        let b3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&d3, &b3, &s3, &s3);
        // update: s_m += W_m @ U (k-loop over C)
        let u_b0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        let w0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[0..1024],
            64u32,
        );
        let w0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1024..2048],
            64u32,
        );
        let w0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2048..3072],
            64u32,
        );
        let w0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3072..4096],
            64u32,
        );
        cmma::execute(&w0_0, &u_b0, &s0, &s0);
        cmma::execute(&w0_1, &u_b0, &s1, &s1);
        cmma::execute(&w0_2, &u_b0, &s2, &s2);
        cmma::execute(&w0_3, &u_b0, &s3, &s3);
        let u_b1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        let w1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[16..1040],
            64u32,
        );
        let w1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1040..2064],
            64u32,
        );
        let w1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2064..3088],
            64u32,
        );
        let w1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3088..4112],
            64u32,
        );
        cmma::execute(&w1_0, &u_b1, &s0, &s0);
        cmma::execute(&w1_1, &u_b1, &s1, &s1);
        cmma::execute(&w1_2, &u_b1, &s2, &s2);
        cmma::execute(&w1_3, &u_b1, &s3, &s3);
        let u_b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        let w2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[32..1056],
            64u32,
        );
        let w2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1056..2080],
            64u32,
        );
        let w2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2080..3104],
            64u32,
        );
        let w2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3104..4128],
            64u32,
        );
        cmma::execute(&w2_0, &u_b2, &s0, &s0);
        cmma::execute(&w2_1, &u_b2, &s1, &s1);
        cmma::execute(&w2_2, &u_b2, &s2, &s2);
        cmma::execute(&w2_3, &u_b2, &s3, &s3);
        let u_b3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        let w3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[48..1072],
            64u32,
        );
        let w3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1072..2096],
            64u32,
        );
        let w3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2096..3120],
            64u32,
        );
        let w3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3120..4144],
            64u32,
        );
        cmma::execute(&w3_0, &u_b3, &s0, &s0);
        cmma::execute(&w3_1, &u_b3, &s1, &s1);
        cmma::execute(&w3_2, &u_b3, &s2, &s2);
        cmma::execute(&w3_3, &u_b3, &s3, &s3);
        // out: o_m = QG_m @ S
        cmma::store(
            s32_sh.as_mut_slice(),
            &s0,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(256usize, 512usize),
            &s1,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(512usize, 768usize),
            &s2,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(768usize, 1024usize),
            &s3,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..KD * VD {
            s16_sh[i] = f16::cast_from(s32_sh[i]);
        }
        sync_cube();
        let o0 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let o1 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let o2 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let o3 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let s_b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        let q0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[0..1024],
            64u32,
        );
        let q1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[1024..2048],
            64u32,
        );
        let q2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2048..3072],
            64u32,
        );
        let q3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[3072..4096],
            64u32,
        );
        cmma::execute(&q0, &s_b2, &o0, &o0);
        cmma::execute(&q1, &s_b2, &o1, &o1);
        cmma::execute(&q2, &s_b2, &o2, &o2);
        cmma::execute(&q3, &s_b2, &o3, &o3);
        let s_b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        let q0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[16..1040],
            64u32,
        );
        let q1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[1040..2064],
            64u32,
        );
        let q2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2064..3088],
            64u32,
        );
        let q3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[3088..4112],
            64u32,
        );
        cmma::execute(&q0, &s_b2, &o0, &o0);
        cmma::execute(&q1, &s_b2, &o1, &o1);
        cmma::execute(&q2, &s_b2, &o2, &o2);
        cmma::execute(&q3, &s_b2, &o3, &o3);
        let s_b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        let q0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[32..1056],
            64u32,
        );
        let q1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[1056..2080],
            64u32,
        );
        let q2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2080..3104],
            64u32,
        );
        let q3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[3104..4128],
            64u32,
        );
        cmma::execute(&q0, &s_b2, &o0, &o0);
        cmma::execute(&q1, &s_b2, &o1, &o1);
        cmma::execute(&q2, &s_b2, &o2, &o2);
        cmma::execute(&q3, &s_b2, &o3, &o3);
        let s_b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        let q0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[48..1072],
            64u32,
        );
        let q1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[1072..2096],
            64u32,
        );
        let q2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2096..3120],
            64u32,
        );
        let q3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[3120..4144],
            64u32,
        );
        cmma::execute(&q0, &s_b2, &o0, &o0);
        cmma::execute(&q1, &s_b2, &o1, &o1);
        cmma::execute(&q2, &s_b2, &o2, &o2);
        cmma::execute(&q3, &s_b2, &o3, &o3);
        let ob = out.slice_mut(hw * out_stride, (hw + 1) * out_stride);
        cmma::store(
            ob.slice_mut(0usize, 256usize),
            &o0,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            ob.slice_mut(256usize, 512usize),
            &o1,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            ob.slice_mut(512usize, 768usize),
            &o2,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            ob.slice_mut(768usize, 1024usize),
            &o3,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
    }
    let sb = state_out.slice_mut(base * KD * VD, (base + 1) * KD * VD);
    cmma::store(
        sb.slice_mut(0usize, 256usize),
        &s0,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        sb.slice_mut(256usize, 512usize),
        &s1,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        sb.slice_mut(512usize, 768usize),
        &s2,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        sb.slice_mut(768usize, 1024usize),
        &s3,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
}

#[test]
fn rr64_matches_cpu() {
    const HEADS: usize = 2;
    const NT: usize = 4;
    let rng = |x: usize| -> f32 { ((x * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5 };
    let mk = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| rng(i) * s).collect() };
    let w_all = mk(HEADS * NT * C * KD, 0.5);
    let u_all = mk(HEADS * NT * C * VD, 0.5);
    let qg_all = mk(HEADS * NT * C * KD, 0.5);
    let g_all: Vec<f32> = (0..HEADS * NT * KD).map(|i| 0.8 + rng(i) * 0.2).collect();
    let state_in = mk(HEADS * KD * VD, 0.2);
    let device = cubecl::cuda::CudaDevice::new(0);
    let client = cubecl::cuda::CudaRuntime::client(&device);
    let mk_buf = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
    let out_h = client.empty(HEADS * NT * C * VD * 4);
    let state_out_h = client.empty(HEADS * KD * VD * 4);
    unsafe {
        rr64_kernel::launch(
            &client,
            CubeCount::Static(HEADS as u32, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            BufferArg::from_raw_parts(mk_buf(&w_all), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&u_all), HEADS * NT * C * VD),
            BufferArg::from_raw_parts(mk_buf(&qg_all), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&g_all), HEADS * NT * KD),
            BufferArg::from_raw_parts(mk_buf(&state_in), HEADS * KD * VD),
            BufferArg::from_raw_parts(state_out_h.clone(), HEADS * KD * VD),
            BufferArg::from_raw_parts(out_h.clone(), HEADS * NT * C * VD),
            NT as u32,
        );
    }
    let raw = client.read_one_unchecked(out_h);
    let got: Vec<f32> = raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let raw_s = client.read_one_unchecked(state_out_h);
    let got_s: Vec<f32> = raw_s
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let mut exp_out = vec![0.0f32; HEADS * NT * C * VD];
    let mut exp_s = vec![0.0f32; HEADS * KD * VD];
    for h in 0..HEADS {
        let mut s = vec![0.0f32; KD * VD];
        for i in 0..KD * VD {
            s[i] = state_in[h * KD * VD + i];
        }
        for j in 0..NT {
            let hw = h * NT + j;
            for r in 0..KD {
                for v in 0..VD {
                    s[r * VD + v] *= g_all[hw * KD + r];
                }
            }
            for r in 0..KD {
                for v in 0..VD {
                    let mut acc = 0.0f32;
                    for k in 0..C {
                        acc += w_all[hw * C * KD + k * KD + r] * u_all[hw * C * VD + k * VD + v];
                    }
                    s[r * VD + v] += acc;
                }
            }
            for t in 0..C {
                for v in 0..VD {
                    let mut acc = 0.0f32;
                    for k in 0..KD {
                        acc += qg_all[hw * C * KD + t * KD + k] * s[k * VD + v];
                    }
                    exp_out[hw * C * VD + t * VD + v] = acc;
                }
            }
        }
        for i in 0..KD * VD {
            exp_s[h * KD * VD + i] = s[i];
        }
    }
    let mut d_out = 0.0f32;
    let mut d_s = 0.0f32;
    for i in 0..exp_out.len() {
        d_out = d_out.max((got[i] - exp_out[i]).abs());
    }
    for i in 0..exp_s.len() {
        d_s = d_s.max((got_s[i] - exp_s[i]).abs());
    }

    assert!(d_out < 2e-1, "rr64 out broken: {d_out:.3e}");
    assert!(d_s < 2e-1, "rr64 state broken: {d_s:.3e}");
}

#[test]
fn rr64_init_only() {
    const HEADS: usize = 2;
    const NT: usize = 0;
    let rng = |x: usize| -> f32 { ((x * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5 };
    let mk = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| rng(i) * s).collect() };
    let w_all = mk(HEADS * 4 * C * KD, 0.5);
    let u_all = mk(HEADS * 4 * C * VD, 0.5);
    let qg_all = mk(HEADS * 4 * C * KD, 0.5);
    let g_all: Vec<f32> = (0..HEADS * 4 * KD).map(|i| 0.8 + rng(i) * 0.2).collect();
    let state_in = mk(HEADS * KD * VD, 0.2);
    let device = cubecl::cuda::CudaDevice::new(0);
    let client = cubecl::cuda::CudaRuntime::client(&device);
    let mk_buf = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
    let out_h = client.empty(HEADS * 4 * C * VD * 4);
    let state_out_h = client.empty(HEADS * KD * VD * 4);
    unsafe {
        rr64_kernel::launch(
            &client,
            CubeCount::Static(HEADS as u32, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            BufferArg::from_raw_parts(mk_buf(&w_all), HEADS * 4 * C * KD),
            BufferArg::from_raw_parts(mk_buf(&u_all), HEADS * 4 * C * VD),
            BufferArg::from_raw_parts(mk_buf(&qg_all), HEADS * 4 * C * KD),
            BufferArg::from_raw_parts(mk_buf(&g_all), HEADS * 4 * KD),
            BufferArg::from_raw_parts(mk_buf(&state_in), HEADS * KD * VD),
            BufferArg::from_raw_parts(state_out_h.clone(), HEADS * KD * VD),
            BufferArg::from_raw_parts(out_h.clone(), HEADS * 4 * C * VD),
            NT as u32,
        );
    }
    let raw_s = client.read_one_unchecked(state_out_h);
    let got_s: Vec<f32> = raw_s
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let mut d = 0.0f32;
    for i in 0..got_s.len() {
        d = d.max((got_s[i] - state_in[i]).abs());
    }

    assert!(d < 1e-3, "init broken: {d:.3e}");
}

#[test]
fn rr64_decay_only() {
    const HEADS: usize = 1;
    const NT: usize = 2;
    let rng = |x: usize| -> f32 { ((x * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5 };
    let w_all = vec![0.0f32; HEADS * NT * C * KD];
    let u_all = vec![0.0f32; HEADS * NT * C * VD];
    let qg_all = vec![0.0f32; HEADS * NT * C * KD];
    let g_all: Vec<f32> = (0..HEADS * NT * KD).map(|i| 0.8 + rng(i) * 0.2).collect();
    let state_in = vec![1.0f32; HEADS * KD * VD];
    let device = cubecl::cuda::CudaDevice::new(0);
    let client = cubecl::cuda::CudaRuntime::client(&device);
    let mk_buf = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
    let out_h = client.empty(HEADS * NT * C * VD * 4);
    let state_out_h = client.empty(HEADS * KD * VD * 4);
    unsafe {
        rr64_kernel::launch(
            &client,
            CubeCount::Static(HEADS as u32, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            BufferArg::from_raw_parts(mk_buf(&w_all), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&u_all), HEADS * NT * C * VD),
            BufferArg::from_raw_parts(mk_buf(&qg_all), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&g_all), HEADS * NT * KD),
            BufferArg::from_raw_parts(mk_buf(&state_in), HEADS * KD * VD),
            BufferArg::from_raw_parts(state_out_h.clone(), HEADS * KD * VD),
            BufferArg::from_raw_parts(out_h.clone(), HEADS * NT * C * VD),
            NT as u32,
        );
    }
    let raw_s = client.read_one_unchecked(state_out_h);
    let got_s: Vec<f32> = raw_s
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let mut exp = vec![1.0f32; HEADS * KD * VD];
    for j in 0..NT {
        for r in 0..KD {
            for v in 0..VD {
                exp[r * VD + v] *= g_all[j * KD + r];
            }
        }
    }
    let mut d = 0.0f32;
    for i in 0..got_s.len() {
        d = d.max((got_s[i] - exp[i]).abs());
    }

    assert!(d < 1e-2, "decay broken: {d:.3e}");
}

#[test]
fn rr64_update_only() {
    const HEADS: usize = 1;
    const NT: usize = 1;
    let rng = |x: usize| -> f32 { ((x * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5 };
    let mk = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| rng(i) * s).collect() };
    let w_all = mk(HEADS * NT * C * KD, 0.5);
    let u_all = mk(HEADS * NT * C * VD, 0.5);
    let qg_all = vec![0.0f32; HEADS * NT * C * KD];
    let g_all = vec![1.0f32; HEADS * NT * KD];
    let state_in = vec![0.0f32; HEADS * KD * VD];
    let device = cubecl::cuda::CudaDevice::new(0);
    let client = cubecl::cuda::CudaRuntime::client(&device);
    let mk_buf = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
    let out_h = client.empty(HEADS * NT * C * VD * 4);
    let state_out_h = client.empty(HEADS * KD * VD * 4);
    unsafe {
        rr64_kernel::launch(
            &client,
            CubeCount::Static(HEADS as u32, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            BufferArg::from_raw_parts(mk_buf(&w_all), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&u_all), HEADS * NT * C * VD),
            BufferArg::from_raw_parts(mk_buf(&qg_all), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&g_all), HEADS * NT * KD),
            BufferArg::from_raw_parts(mk_buf(&state_in), HEADS * KD * VD),
            BufferArg::from_raw_parts(state_out_h.clone(), HEADS * KD * VD),
            BufferArg::from_raw_parts(out_h.clone(), HEADS * NT * C * VD),
            NT as u32,
        );
    }
    let raw_s = client.read_one_unchecked(state_out_h);
    let got_s: Vec<f32> = raw_s
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let mut exp = vec![0.0f32; HEADS * KD * VD];
    for r in 0..KD {
        for v in 0..VD {
            let mut acc = 0.0f32;
            for k in 0..C {
                acc += w_all[k * KD + r] * u_all[k * VD + v];
            }
            exp[r * VD + v] = acc;
        }
    }
    let mut d = 0.0f32;
    for i in 0..got_s.len() {
        d = d.max((got_s[i] - exp[i]).abs());
    }
    // sanity: w=u=1 -> S[r][v] = C = 64
    let w1 = vec![1.0f32; HEADS * NT * C * KD];
    let u1 = vec![1.0f32; HEADS * NT * C * VD];
    let state_out_h2 = client.empty(HEADS * KD * VD * 4);
    let out_h2 = client.empty(HEADS * NT * C * VD * 4);
    unsafe {
        rr64_kernel::launch(
            &client,
            CubeCount::Static(HEADS as u32, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            BufferArg::from_raw_parts(mk_buf(&w1), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&u1), HEADS * NT * C * VD),
            BufferArg::from_raw_parts(mk_buf(&qg_all), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&g_all), HEADS * NT * KD),
            BufferArg::from_raw_parts(mk_buf(&state_in), HEADS * KD * VD),
            BufferArg::from_raw_parts(state_out_h2.clone(), HEADS * KD * VD),
            BufferArg::from_raw_parts(out_h2.clone(), HEADS * NT * C * VD),
            NT as u32,
        );
    }
    let raw_s2 = client.read_one_unchecked(state_out_h2);
    let got_s2: Vec<f32> = raw_s2
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    println!(
        "  w=u=1 sanity: s[0]={:.3} s[1]={:.3} s[16]={:.3} s[17]={:.3} (expect 64)",
        got_s2[0], got_s2[1], got_s2[16], got_s2[17]
    );

    assert!(d < 2e-2, "update broken: {d:.3e}");
}