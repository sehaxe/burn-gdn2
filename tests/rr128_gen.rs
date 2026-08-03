//! Full-scale register-resident recurrence: kd=128, vd=128, c=64.
//! Grid (heads, n-tiles, 1); one warp per (head, v-slice of 16).
//! State lives in 8 CMMA accumulators (registers) across chunks; decay,
//! update and out all go through tensor cores.

use cubecl::cmma;
use cubecl::prelude::*;
use half::f16;

const KD: usize = 128;
const C: usize = 64;

#[cube(launch)]
fn rr128_kernel(
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
    let ntile = CUBE_POS_Y as usize;
    let n0 = ntile * 16usize;
    let w_stride = C * KD;
    let u_stride = C * 128;
    let qg_stride = C * KD;
    let out_stride = C * 128;
    let base = head;

    let mut w_sh = Shared::<[f16]>::new_slice(C * KD);
    let mut u_sh = Shared::<[f16]>::new_slice(C * 16);
    let mut qg_sh = Shared::<[f16]>::new_slice(C * KD);
    let mut s32_sh = Shared::<[f32]>::new_slice(KD * 16);
    let mut s16_sh = Shared::<[f16]>::new_slice(KD * 16);
    let mut o32_sh = Shared::<[f32]>::new_slice(C * 16);
    let mut eye16_sh = Shared::<[f16]>::new_slice(256usize);
    let mut diag16_sh = Shared::<[f16]>::new_slice(256usize);

    for i in 0..256usize {
        eye16_sh[i] = f16::cast_from(0.0f32);
    }
    for i in 0..16usize {
        eye16_sh[i * 16 + i] = f16::cast_from(1.0f32);
    }
    // state in -> s32_sh (stride unpack: state[k][n0+v])
    for i in 0..KD * 16 {
        s32_sh[i] = state_in[base * KD * 128 + (i / 16) * 128 + n0 + (i % 16)];
    }
    sync_cube();
    for i in 0..KD * 16 {
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
    let s0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b0, &s0, &s0);
    let b1 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[256..512],
        16u32,
    );
    let s1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b1, &s1, &s1);
    let b2 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[512..768],
        16u32,
    );
    let s2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b2, &s2, &s2);
    let b3 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[768..1024],
        16u32,
    );
    let s3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b3, &s3, &s3);
    let b4 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[1024..1280],
        16u32,
    );
    let s4 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b4, &s4, &s4);
    let b5 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[1280..1536],
        16u32,
    );
    let s5 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b5, &s5, &s5);
    let b6 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[1536..1792],
        16u32,
    );
    let s6 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b6, &s6, &s6);
    let b7 = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &s16_sh[1792..2048],
        16u32,
    );
    let s7 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );
    cmma::execute(&eye, &b7, &s7, &s7);
    for j in range_stepped(0u32, nt, 1u32) {
        let jj = j as usize;
        let hw = base * (nt as usize) + jj;
        for i in 0..C * KD {
            let t = i / KD;
            let kd = i % KD;
            w_sh[kd * C + t] = f16::cast_from(w_all[hw * w_stride + i]);
        }
        for i in 0..C * 16 {
            let t = i / 16;
            let v = i % 16;
            u_sh[t * 16 + v] = f16::cast_from(u_all[hw * u_stride + t * 128 + n0 + v]);
        }
        for i in 0..C * KD {
            qg_sh[i] = f16::cast_from(qg_all[hw * qg_stride + i]);
        }
        sync_cube();
        // S -> shared (f16)
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(0usize, 256usize),
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
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1024usize, 1280usize),
            &s4,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1280usize, 1536usize),
            &s5,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1536usize, 1792usize),
            &s6,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1792usize, 2048usize),
            &s7,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..KD * 16 {
            s16_sh[i] = f16::cast_from(s32_sh[i]);
        }
        sync_cube();
        // decay: s_{kk} += (diag16_kk - I) @ S_{kk}
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] = f16::cast_from(g_all[hw * KD + i]) - f16::cast_from(1.0f32);
        }
        let d0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&d0, &sb0, &s0, &s0);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 16 + i]) - f16::cast_from(1.0f32);
        }
        let d1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&d1, &sb1, &s1, &s1);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 32 + i]) - f16::cast_from(1.0f32);
        }
        let d2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&d2, &sb2, &s2, &s2);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 48 + i]) - f16::cast_from(1.0f32);
        }
        let d3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&d3, &sb3, &s3, &s3);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 64 + i]) - f16::cast_from(1.0f32);
        }
        let d4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&d4, &sb4, &s4, &s4);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 80 + i]) - f16::cast_from(1.0f32);
        }
        let d5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&d5, &sb5, &s5, &s5);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 96 + i]) - f16::cast_from(1.0f32);
        }
        let d6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&d6, &sb6, &s6, &s6);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(g_all[hw * KD + 112 + i]) - f16::cast_from(1.0f32);
        }
        let d7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &diag16_sh[0..256],
            16u32,
        );
        let sb7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&d7, &sb7, &s7, &s7);
        // update: s_{kk} += W^T_{kk} @ U (W транспонирован в shared; k-цикл по C)
        let w0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[0..1024],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_0, &u0, &s0, &s0);
        let w0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1024..2048],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_1, &u0, &s1, &s1);
        let w0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2048..3072],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_2, &u0, &s2, &s2);
        let w0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3072..4096],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_3, &u0, &s3, &s3);
        let w0_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4096..5120],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_4, &u0, &s4, &s4);
        let w0_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[5120..6144],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_5, &u0, &s5, &s5);
        let w0_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6144..7168],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_6, &u0, &s6, &s6);
        let w0_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[7168..8192],
            64u32,
        );
        let u0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[0..256],
            16u32,
        );
        cmma::execute(&w0_7, &u0, &s7, &s7);
        let w1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[16..1040],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_0, &u1, &s0, &s0);
        let w1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1040..2064],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_1, &u1, &s1, &s1);
        let w1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2064..3088],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_2, &u1, &s2, &s2);
        let w1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3088..4112],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_3, &u1, &s3, &s3);
        let w1_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4112..5136],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_4, &u1, &s4, &s4);
        let w1_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[5136..6160],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_5, &u1, &s5, &s5);
        let w1_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6160..7184],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_6, &u1, &s6, &s6);
        let w1_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[7184..8208],
            64u32,
        );
        let u1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[256..512],
            16u32,
        );
        cmma::execute(&w1_7, &u1, &s7, &s7);
        let w2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[32..1056],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_0, &u2, &s0, &s0);
        let w2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1056..2080],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_1, &u2, &s1, &s1);
        let w2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2080..3104],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_2, &u2, &s2, &s2);
        let w2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3104..4128],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_3, &u2, &s3, &s3);
        let w2_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4128..5152],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_4, &u2, &s4, &s4);
        let w2_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[5152..6176],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_5, &u2, &s5, &s5);
        let w2_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6176..7200],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_6, &u2, &s6, &s6);
        let w2_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[7200..8224],
            64u32,
        );
        let u2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[512..768],
            16u32,
        );
        cmma::execute(&w2_7, &u2, &s7, &s7);
        let w3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[48..1072],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_0, &u3, &s0, &s0);
        let w3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[1072..2096],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_1, &u3, &s1, &s1);
        let w3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2096..3120],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_2, &u3, &s2, &s2);
        let w3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[3120..4144],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_3, &u3, &s3, &s3);
        let w3_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4144..5168],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_4, &u3, &s4, &s4);
        let w3_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[5168..6192],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_5, &u3, &s5, &s5);
        let w3_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6192..7216],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_6, &u3, &s6, &s6);
        let w3_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[7216..8240],
            64u32,
        );
        let u3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &u_sh[768..1024],
            16u32,
        );
        cmma::execute(&w3_7, &u3, &s7, &s7);
        // S -> shared (f16) для out
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(0usize, 256usize),
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
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1024usize, 1280usize),
            &s4,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1280usize, 1536usize),
            &s5,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1536usize, 1792usize),
            &s6,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            s32_sh.as_mut_slice().slice_mut(1792usize, 2048usize),
            &s7,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..KD * 16 {
            s16_sh[i] = f16::cast_from(s32_sh[i]);
        }
        sync_cube();
        // out: o_m = QG_m @ S  (m по токенам: C/16 штук; kk по kd: MT штук)
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
        let q0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[0..2048],
            128u32,
        );
        let ob0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&q0_0, &ob0_0, &o0, &o0);
        let q0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2048..4096],
            128u32,
        );
        let ob0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&q0_1, &ob0_1, &o1, &o1);
        let q0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4096..6144],
            128u32,
        );
        let ob0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&q0_2, &ob0_2, &o2, &o2);
        let q0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6144..8192],
            128u32,
        );
        let ob0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&q0_3, &ob0_3, &o3, &o3);
        let q1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[16..2064],
            128u32,
        );
        let ob1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&q1_0, &ob1_0, &o0, &o0);
        let q1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2064..4112],
            128u32,
        );
        let ob1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&q1_1, &ob1_1, &o1, &o1);
        let q1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4112..6160],
            128u32,
        );
        let ob1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&q1_2, &ob1_2, &o2, &o2);
        let q1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6160..8208],
            128u32,
        );
        let ob1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&q1_3, &ob1_3, &o3, &o3);
        let q2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[32..2080],
            128u32,
        );
        let ob2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&q2_0, &ob2_0, &o0, &o0);
        let q2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2080..4128],
            128u32,
        );
        let ob2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&q2_1, &ob2_1, &o1, &o1);
        let q2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4128..6176],
            128u32,
        );
        let ob2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&q2_2, &ob2_2, &o2, &o2);
        let q2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6176..8224],
            128u32,
        );
        let ob2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&q2_3, &ob2_3, &o3, &o3);
        let q3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[48..2096],
            128u32,
        );
        let ob3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&q3_0, &ob3_0, &o0, &o0);
        let q3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2096..4144],
            128u32,
        );
        let ob3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&q3_1, &ob3_1, &o1, &o1);
        let q3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4144..6192],
            128u32,
        );
        let ob3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&q3_2, &ob3_2, &o2, &o2);
        let q3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6192..8240],
            128u32,
        );
        let ob3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&q3_3, &ob3_3, &o3, &o3);
        let q4_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[64..2112],
            128u32,
        );
        let ob4_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&q4_0, &ob4_0, &o0, &o0);
        let q4_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2112..4160],
            128u32,
        );
        let ob4_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&q4_1, &ob4_1, &o1, &o1);
        let q4_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4160..6208],
            128u32,
        );
        let ob4_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&q4_2, &ob4_2, &o2, &o2);
        let q4_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6208..8256],
            128u32,
        );
        let ob4_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&q4_3, &ob4_3, &o3, &o3);
        let q5_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[80..2128],
            128u32,
        );
        let ob5_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&q5_0, &ob5_0, &o0, &o0);
        let q5_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2128..4176],
            128u32,
        );
        let ob5_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&q5_1, &ob5_1, &o1, &o1);
        let q5_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4176..6224],
            128u32,
        );
        let ob5_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&q5_2, &ob5_2, &o2, &o2);
        let q5_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6224..8272],
            128u32,
        );
        let ob5_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&q5_3, &ob5_3, &o3, &o3);
        let q6_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[96..2144],
            128u32,
        );
        let ob6_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&q6_0, &ob6_0, &o0, &o0);
        let q6_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2144..4192],
            128u32,
        );
        let ob6_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&q6_1, &ob6_1, &o1, &o1);
        let q6_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4192..6240],
            128u32,
        );
        let ob6_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&q6_2, &ob6_2, &o2, &o2);
        let q6_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6240..8288],
            128u32,
        );
        let ob6_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&q6_3, &ob6_3, &o3, &o3);
        let q7_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[112..2160],
            128u32,
        );
        let ob7_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&q7_0, &ob7_0, &o0, &o0);
        let q7_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2160..4208],
            128u32,
        );
        let ob7_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&q7_1, &ob7_1, &o1, &o1);
        let q7_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4208..6256],
            128u32,
        );
        let ob7_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&q7_2, &ob7_2, &o2, &o2);
        let q7_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6256..8304],
            128u32,
        );
        let ob7_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&q7_3, &ob7_3, &o3, &o3);
        // out -> global (stride pack)
        cmma::store(
            o32_sh.as_mut_slice(),
            &o0,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            o32_sh.as_mut_slice().slice_mut(256usize, 512usize),
            &o1,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            o32_sh.as_mut_slice().slice_mut(512usize, 768usize),
            &o2,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            o32_sh.as_mut_slice().slice_mut(768usize, 1024usize),
            &o3,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..C * 16 {
            out[hw * out_stride + (i / 16) * 128 + n0 + (i % 16)] = o32_sh[i];
        }
        sync_cube();
    }
    // state_out (stride pack)
    let sb = state_out.slice_mut(base * KD * 128, (base + 1) * KD * 128);
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
    cmma::store(
        s32_sh.as_mut_slice().slice_mut(1024usize, 1280usize),
        &s4,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        s32_sh.as_mut_slice().slice_mut(1280usize, 1536usize),
        &s5,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        s32_sh.as_mut_slice().slice_mut(1536usize, 1792usize),
        &s6,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        s32_sh.as_mut_slice().slice_mut(1792usize, 2048usize),
        &s7,
        16u32,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    for i in 0..KD * 16 {
        sb[(i / 16) * 128 + n0 + (i % 16)] = s32_sh[i];
    }
}
#[test]
fn rr128_sanity() {
    const HEADS: usize = 1;
    const NT: usize = 1;
    const VD: usize = 128;
    const KD: usize = 128;
    const C: usize = 64;
    let device = cubecl::cuda::CudaDevice::new(0);
    let client = cubecl::cuda::CudaRuntime::client(&device);
    let mk_buf = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
    let w0 = vec![0.0f32; HEADS * NT * C * KD];
    let u0 = vec![0.0f32; HEADS * NT * C * VD];
    let qg1 = vec![1.0f32; HEADS * NT * C * KD];
    let g1 = vec![1.0f32; HEADS * NT * KD];
    let st1 = vec![1.0f32; HEADS * KD * VD];
    let out_h = client.empty(HEADS * NT * C * VD * 4);
    let state_out_h = client.empty(HEADS * KD * VD * 4);
    unsafe {
        rr128_kernel::launch(
            &client,
            CubeCount::Static(HEADS as u32, (VD / 16) as u32, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            BufferArg::from_raw_parts(mk_buf(&w0), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&u0), HEADS * NT * C * VD),
            BufferArg::from_raw_parts(mk_buf(&qg1), HEADS * NT * C * KD),
            BufferArg::from_raw_parts(mk_buf(&g1), HEADS * NT * KD),
            BufferArg::from_raw_parts(mk_buf(&st1), HEADS * KD * VD),
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
    let d: f32 = got.iter().map(|x| (x - 128.0).abs()).fold(0.0f32, f32::max);
    assert!(d < 1.0, "sanity broken: {d:.3e}");
}

#[test]
fn rr128_matches_cpu() {
    const HEADS: usize = 2;
    const NT: usize = 2;
    const VD: usize = 128;
    const KD: usize = 128;
    const C: usize = 64;
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
        rr128_kernel::launch(
            &client,
            CubeCount::Static(HEADS as u32, (VD / 16) as u32, 1),
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
    assert!(d_out < 2e-1, "rr128 out broken: {d_out:.3e}");
    assert!(d_s < 2e-1, "rr128 state broken: {d_s:.3e}");
}
