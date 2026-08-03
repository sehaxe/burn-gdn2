//! Register-resident inter recurrence with the REAL GDN-2 step (matches
//! `gdn2_chunk_inter_kernel` math): vn = u - w@S, out = aqk@vn + scale*qg@S,
//! S = diag(glast)@S + kgd^T@vn. All dots via tensor cores (f16, f32 acc).
//! One warp per (head, v-slice of 16); S in 8 accumulators.

use cubecl::cmma;
use cubecl::prelude::*;
use half::f16;

const KD: usize = 128;
const C: usize = 64;

#[cube(launch)]
fn rr_inter_kernel(
    w_all: &[f32],     // [BH*NT, KD, C] транспонированный: w[k*C + r]
    u_all: &[f32],     // [BH*NT, C, VD] row-major
    qg_all: &[f32],    // [BH*NT, KD, C] транспонированный: qg[k*C + r]
    kgd_all: &[f32],   // [BH*NT, KD, C] транспонированный
    aqk_all: &[f32],   // [BH*NT, C, C] row-major
    glast_all: &[f32], // [BH*NT, KD]
    state_in: &[f32],  // [BH, KD, VD]
    state_out: &mut [f32],
    out: &mut [f32], // [BH*NT, C, VD]
    nt: u32,
    scale: f32,
) {
    let head = CUBE_POS_X as usize;
    let ntile = CUBE_POS_Y as usize;
    let n0 = ntile * 16usize;
    let u_stride = C * 128;
    let out_stride = C * 128;
    let base = head;

    let mut w_sh = Shared::<[f16]>::new_slice(C * KD);
    let mut u_sh = Shared::<[f16]>::new_slice(C * 16);
    let mut qg_sh = Shared::<[f16]>::new_slice(C * KD);
    let mut kgd_sh = Shared::<[f16]>::new_slice(KD * C);
    let mut aqk_sh = Shared::<[f16]>::new_slice(C * C);
    let mut s32_sh = Shared::<[f32]>::new_slice(KD * 16);
    let mut s16_sh = Shared::<[f16]>::new_slice(KD * 16);
    let mut vn32_sh = Shared::<[f32]>::new_slice(C * 16);
    let mut vn16_sh = Shared::<[f16]>::new_slice(C * 16);
    let mut o32_sh = Shared::<[f32]>::new_slice(C * 16);
    let mut eye16_sh = Shared::<[f16]>::new_slice(256usize);
    let mut diag16_sh = Shared::<[f16]>::new_slice(256usize);

    for i in 0..256usize {
        eye16_sh[i] = f16::cast_from(0.0f32);
    }
    for i in 0..16usize {
        eye16_sh[i * 16 + i] = f16::cast_from(1.0f32);
    }
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
            w_sh[t * KD + kd] = f16::cast_from(w_all[hw * KD * C + kd * C + t]);
            qg_sh[t * KD + kd] =
                f16::cast_from(qg_all[hw * KD * C + kd * C + t]) * f16::cast_from(scale);
        }
        for i in 0..C * 16 {
            let t = i / 16;
            let v = i % 16;
            u_sh[t * 16 + v] = f16::cast_from(u_all[hw * u_stride + t * 128 + n0 + v]);
        }
        for i in 0..C * KD {
            let t = i / KD;
            let kd = i % KD;
            kgd_sh[kd * C + t] = f16::cast_from(kgd_all[hw * C * KD + t * KD + kd]);
        }
        for i in 0..C * C {
            let r = i / C;
            let s = i % C;
            aqk_sh[r * C + s] = f16::cast_from(aqk_all[hw * C * C + s * C + r]);
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
        // vn = w @ S
        let vn0 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let vn1 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let vn2 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let vn3 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        let wm0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[0..2048],
            128u32,
        );
        let sb0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&wm0_0, &sb0_0, &vn0, &vn0);
        let wm0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2048..4096],
            128u32,
        );
        let sb0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&wm0_1, &sb0_1, &vn1, &vn1);
        let wm0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4096..6144],
            128u32,
        );
        let sb0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&wm0_2, &sb0_2, &vn2, &vn2);
        let wm0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6144..8192],
            128u32,
        );
        let sb0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&wm0_3, &sb0_3, &vn3, &vn3);
        let wm1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[16..2064],
            128u32,
        );
        let sb1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&wm1_0, &sb1_0, &vn0, &vn0);
        let wm1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2064..4112],
            128u32,
        );
        let sb1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&wm1_1, &sb1_1, &vn1, &vn1);
        let wm1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4112..6160],
            128u32,
        );
        let sb1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&wm1_2, &sb1_2, &vn2, &vn2);
        let wm1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6160..8208],
            128u32,
        );
        let sb1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&wm1_3, &sb1_3, &vn3, &vn3);
        let wm2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[32..2080],
            128u32,
        );
        let sb2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&wm2_0, &sb2_0, &vn0, &vn0);
        let wm2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2080..4128],
            128u32,
        );
        let sb2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&wm2_1, &sb2_1, &vn1, &vn1);
        let wm2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4128..6176],
            128u32,
        );
        let sb2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&wm2_2, &sb2_2, &vn2, &vn2);
        let wm2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6176..8224],
            128u32,
        );
        let sb2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&wm2_3, &sb2_3, &vn3, &vn3);
        let wm3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[48..2096],
            128u32,
        );
        let sb3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&wm3_0, &sb3_0, &vn0, &vn0);
        let wm3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2096..4144],
            128u32,
        );
        let sb3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&wm3_1, &sb3_1, &vn1, &vn1);
        let wm3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4144..6192],
            128u32,
        );
        let sb3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&wm3_2, &sb3_2, &vn2, &vn2);
        let wm3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6192..8240],
            128u32,
        );
        let sb3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&wm3_3, &sb3_3, &vn3, &vn3);
        let wm4_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[64..2112],
            128u32,
        );
        let sb4_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&wm4_0, &sb4_0, &vn0, &vn0);
        let wm4_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2112..4160],
            128u32,
        );
        let sb4_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&wm4_1, &sb4_1, &vn1, &vn1);
        let wm4_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4160..6208],
            128u32,
        );
        let sb4_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&wm4_2, &sb4_2, &vn2, &vn2);
        let wm4_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6208..8256],
            128u32,
        );
        let sb4_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&wm4_3, &sb4_3, &vn3, &vn3);
        let wm5_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[80..2128],
            128u32,
        );
        let sb5_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&wm5_0, &sb5_0, &vn0, &vn0);
        let wm5_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2128..4176],
            128u32,
        );
        let sb5_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&wm5_1, &sb5_1, &vn1, &vn1);
        let wm5_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4176..6224],
            128u32,
        );
        let sb5_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&wm5_2, &sb5_2, &vn2, &vn2);
        let wm5_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6224..8272],
            128u32,
        );
        let sb5_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&wm5_3, &sb5_3, &vn3, &vn3);
        let wm6_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[96..2144],
            128u32,
        );
        let sb6_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&wm6_0, &sb6_0, &vn0, &vn0);
        let wm6_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2144..4192],
            128u32,
        );
        let sb6_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&wm6_1, &sb6_1, &vn1, &vn1);
        let wm6_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4192..6240],
            128u32,
        );
        let sb6_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&wm6_2, &sb6_2, &vn2, &vn2);
        let wm6_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6240..8288],
            128u32,
        );
        let sb6_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&wm6_3, &sb6_3, &vn3, &vn3);
        let wm7_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[112..2160],
            128u32,
        );
        let sb7_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&wm7_0, &sb7_0, &vn0, &vn0);
        let wm7_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[2160..4208],
            128u32,
        );
        let sb7_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&wm7_1, &sb7_1, &vn1, &vn1);
        let wm7_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[4208..6256],
            128u32,
        );
        let sb7_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&wm7_2, &sb7_2, &vn2, &vn2);
        let wm7_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &w_sh[6256..8304],
            128u32,
        );
        let sb7_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&wm7_3, &sb7_3, &vn3, &vn3);
        // vn -> u - vn (shared elementwise)
        cmma::store(
            vn32_sh.as_mut_slice(),
            &vn0,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            vn32_sh.as_mut_slice().slice_mut(256usize, 512usize),
            &vn1,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            vn32_sh.as_mut_slice().slice_mut(512usize, 768usize),
            &vn2,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            vn32_sh.as_mut_slice().slice_mut(768usize, 1024usize),
            &vn3,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        for i in 0..C * 16 {
            vn16_sh[i] = f16::cast_from(u_sh[i]) - f16::cast_from(vn32_sh[i]);
        }
        sync_cube();
        // out = scale*qg @ S + aqk @ vn
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
        let qm0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[0..2048],
            128u32,
        );
        let sb20_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&qm0_0, &sb20_0, &o0, &o0);
        let qm0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2048..4096],
            128u32,
        );
        let sb20_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&qm0_1, &sb20_1, &o1, &o1);
        let qm0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4096..6144],
            128u32,
        );
        let sb20_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&qm0_2, &sb20_2, &o2, &o2);
        let qm0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6144..8192],
            128u32,
        );
        let sb20_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&qm0_3, &sb20_3, &o3, &o3);
        let qm1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[16..2064],
            128u32,
        );
        let sb21_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&qm1_0, &sb21_0, &o0, &o0);
        let qm1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2064..4112],
            128u32,
        );
        let sb21_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&qm1_1, &sb21_1, &o1, &o1);
        let qm1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4112..6160],
            128u32,
        );
        let sb21_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&qm1_2, &sb21_2, &o2, &o2);
        let qm1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6160..8208],
            128u32,
        );
        let sb21_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&qm1_3, &sb21_3, &o3, &o3);
        let qm2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[32..2080],
            128u32,
        );
        let sb22_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&qm2_0, &sb22_0, &o0, &o0);
        let qm2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2080..4128],
            128u32,
        );
        let sb22_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&qm2_1, &sb22_1, &o1, &o1);
        let qm2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4128..6176],
            128u32,
        );
        let sb22_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&qm2_2, &sb22_2, &o2, &o2);
        let qm2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6176..8224],
            128u32,
        );
        let sb22_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&qm2_3, &sb22_3, &o3, &o3);
        let qm3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[48..2096],
            128u32,
        );
        let sb23_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&qm3_0, &sb23_0, &o0, &o0);
        let qm3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2096..4144],
            128u32,
        );
        let sb23_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&qm3_1, &sb23_1, &o1, &o1);
        let qm3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4144..6192],
            128u32,
        );
        let sb23_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&qm3_2, &sb23_2, &o2, &o2);
        let qm3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6192..8240],
            128u32,
        );
        let sb23_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&qm3_3, &sb23_3, &o3, &o3);
        let qm4_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[64..2112],
            128u32,
        );
        let sb24_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&qm4_0, &sb24_0, &o0, &o0);
        let qm4_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2112..4160],
            128u32,
        );
        let sb24_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&qm4_1, &sb24_1, &o1, &o1);
        let qm4_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4160..6208],
            128u32,
        );
        let sb24_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&qm4_2, &sb24_2, &o2, &o2);
        let qm4_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6208..8256],
            128u32,
        );
        let sb24_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&qm4_3, &sb24_3, &o3, &o3);
        let qm5_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[80..2128],
            128u32,
        );
        let sb25_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&qm5_0, &sb25_0, &o0, &o0);
        let qm5_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2128..4176],
            128u32,
        );
        let sb25_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&qm5_1, &sb25_1, &o1, &o1);
        let qm5_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4176..6224],
            128u32,
        );
        let sb25_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&qm5_2, &sb25_2, &o2, &o2);
        let qm5_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6224..8272],
            128u32,
        );
        let sb25_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&qm5_3, &sb25_3, &o3, &o3);
        let qm6_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[96..2144],
            128u32,
        );
        let sb26_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&qm6_0, &sb26_0, &o0, &o0);
        let qm6_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2144..4192],
            128u32,
        );
        let sb26_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&qm6_1, &sb26_1, &o1, &o1);
        let qm6_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4192..6240],
            128u32,
        );
        let sb26_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&qm6_2, &sb26_2, &o2, &o2);
        let qm6_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6240..8288],
            128u32,
        );
        let sb26_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&qm6_3, &sb26_3, &o3, &o3);
        let qm7_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[112..2160],
            128u32,
        );
        let sb27_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&qm7_0, &sb27_0, &o0, &o0);
        let qm7_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[2160..4208],
            128u32,
        );
        let sb27_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&qm7_1, &sb27_1, &o1, &o1);
        let qm7_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[4208..6256],
            128u32,
        );
        let sb27_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&qm7_2, &sb27_2, &o2, &o2);
        let qm7_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &qg_sh[6256..8304],
            128u32,
        );
        let sb27_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&qm7_3, &sb27_3, &o3, &o3);
        let am0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[0..1024],
            64u32,
        );
        let vb0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&am0_0, &vb0_0, &o0, &o0);
        let am0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[1024..2048],
            64u32,
        );
        let vb0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&am0_1, &vb0_1, &o1, &o1);
        let am0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[2048..3072],
            64u32,
        );
        let vb0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&am0_2, &vb0_2, &o2, &o2);
        let am0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[3072..4096],
            64u32,
        );
        let vb0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&am0_3, &vb0_3, &o3, &o3);
        let am1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[16..1040],
            64u32,
        );
        let vb1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&am1_0, &vb1_0, &o0, &o0);
        let am1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[1040..2064],
            64u32,
        );
        let vb1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&am1_1, &vb1_1, &o1, &o1);
        let am1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[2064..3088],
            64u32,
        );
        let vb1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&am1_2, &vb1_2, &o2, &o2);
        let am1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[3088..4112],
            64u32,
        );
        let vb1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&am1_3, &vb1_3, &o3, &o3);
        let am2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[32..1056],
            64u32,
        );
        let vb2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&am2_0, &vb2_0, &o0, &o0);
        let am2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[1056..2080],
            64u32,
        );
        let vb2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&am2_1, &vb2_1, &o1, &o1);
        let am2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[2080..3104],
            64u32,
        );
        let vb2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&am2_2, &vb2_2, &o2, &o2);
        let am2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[3104..4128],
            64u32,
        );
        let vb2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&am2_3, &vb2_3, &o3, &o3);
        let am3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[48..1072],
            64u32,
        );
        let vb3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&am3_0, &vb3_0, &o0, &o0);
        let am3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[1072..2096],
            64u32,
        );
        let vb3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&am3_1, &vb3_1, &o1, &o1);
        let am3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[2096..3120],
            64u32,
        );
        let vb3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&am3_2, &vb3_2, &o2, &o2);
        let am3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &aqk_sh[3120..4144],
            64u32,
        );
        let vb3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&am3_3, &vb3_3, &o3, &o3);
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
        // decay: S_m += (diag16_m - I) @ S_m
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] = f16::cast_from(glast_all[hw * KD + i]) - f16::cast_from(1.0f32);
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
        let sd0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[0..256],
            16u32,
        );
        cmma::execute(&d0, &sd0, &s0, &s0);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(glast_all[hw * KD + 16 + i]) - f16::cast_from(1.0f32);
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
        let sd1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[256..512],
            16u32,
        );
        cmma::execute(&d1, &sd1, &s1, &s1);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(glast_all[hw * KD + 32 + i]) - f16::cast_from(1.0f32);
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
        let sd2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[512..768],
            16u32,
        );
        cmma::execute(&d2, &sd2, &s2, &s2);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(glast_all[hw * KD + 48 + i]) - f16::cast_from(1.0f32);
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
        let sd3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[768..1024],
            16u32,
        );
        cmma::execute(&d3, &sd3, &s3, &s3);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(glast_all[hw * KD + 64 + i]) - f16::cast_from(1.0f32);
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
        let sd4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1024..1280],
            16u32,
        );
        cmma::execute(&d4, &sd4, &s4, &s4);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(glast_all[hw * KD + 80 + i]) - f16::cast_from(1.0f32);
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
        let sd5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1280..1536],
            16u32,
        );
        cmma::execute(&d5, &sd5, &s5, &s5);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(glast_all[hw * KD + 96 + i]) - f16::cast_from(1.0f32);
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
        let sd6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1536..1792],
            16u32,
        );
        cmma::execute(&d6, &sd6, &s6, &s6);
        for i in 0..256usize {
            diag16_sh[i] = f16::cast_from(0.0f32);
        }
        for i in 0..16usize {
            diag16_sh[i * 16 + i] =
                f16::cast_from(glast_all[hw * KD + 112 + i]) - f16::cast_from(1.0f32);
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
        let sd7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &s16_sh[1792..2048],
            16u32,
        );
        cmma::execute(&d7, &sd7, &s7, &s7);
        // S_m += kgd^T_m @ vn
        let km0_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[0..1024],
            64u32,
        );
        let vb20_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_0, &vb20_0, &s0, &s0);
        let km0_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[1024..2048],
            64u32,
        );
        let vb20_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_1, &vb20_1, &s1, &s1);
        let km0_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[2048..3072],
            64u32,
        );
        let vb20_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_2, &vb20_2, &s2, &s2);
        let km0_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[3072..4096],
            64u32,
        );
        let vb20_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_3, &vb20_3, &s3, &s3);
        let km0_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[4096..5120],
            64u32,
        );
        let vb20_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_4, &vb20_4, &s4, &s4);
        let km0_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[5120..6144],
            64u32,
        );
        let vb20_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_5, &vb20_5, &s5, &s5);
        let km0_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[6144..7168],
            64u32,
        );
        let vb20_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_6, &vb20_6, &s6, &s6);
        let km0_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[7168..8192],
            64u32,
        );
        let vb20_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[0..256],
            16u32,
        );
        cmma::execute(&km0_7, &vb20_7, &s7, &s7);
        let km1_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[16..1040],
            64u32,
        );
        let vb21_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_0, &vb21_0, &s0, &s0);
        let km1_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[1040..2064],
            64u32,
        );
        let vb21_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_1, &vb21_1, &s1, &s1);
        let km1_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[2064..3088],
            64u32,
        );
        let vb21_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_2, &vb21_2, &s2, &s2);
        let km1_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[3088..4112],
            64u32,
        );
        let vb21_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_3, &vb21_3, &s3, &s3);
        let km1_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[4112..5136],
            64u32,
        );
        let vb21_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_4, &vb21_4, &s4, &s4);
        let km1_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[5136..6160],
            64u32,
        );
        let vb21_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_5, &vb21_5, &s5, &s5);
        let km1_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[6160..7184],
            64u32,
        );
        let vb21_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_6, &vb21_6, &s6, &s6);
        let km1_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[7184..8208],
            64u32,
        );
        let vb21_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[256..512],
            16u32,
        );
        cmma::execute(&km1_7, &vb21_7, &s7, &s7);
        let km2_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[32..1056],
            64u32,
        );
        let vb22_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_0, &vb22_0, &s0, &s0);
        let km2_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[1056..2080],
            64u32,
        );
        let vb22_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_1, &vb22_1, &s1, &s1);
        let km2_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[2080..3104],
            64u32,
        );
        let vb22_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_2, &vb22_2, &s2, &s2);
        let km2_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[3104..4128],
            64u32,
        );
        let vb22_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_3, &vb22_3, &s3, &s3);
        let km2_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[4128..5152],
            64u32,
        );
        let vb22_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_4, &vb22_4, &s4, &s4);
        let km2_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[5152..6176],
            64u32,
        );
        let vb22_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_5, &vb22_5, &s5, &s5);
        let km2_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[6176..7200],
            64u32,
        );
        let vb22_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_6, &vb22_6, &s6, &s6);
        let km2_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[7200..8224],
            64u32,
        );
        let vb22_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[512..768],
            16u32,
        );
        cmma::execute(&km2_7, &vb22_7, &s7, &s7);
        let km3_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[48..1072],
            64u32,
        );
        let vb23_0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_0, &vb23_0, &s0, &s0);
        let km3_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[1072..2096],
            64u32,
        );
        let vb23_1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_1, &vb23_1, &s1, &s1);
        let km3_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[2096..3120],
            64u32,
        );
        let vb23_2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_2, &vb23_2, &s2, &s2);
        let km3_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[3120..4144],
            64u32,
        );
        let vb23_3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_3, &vb23_3, &s3, &s3);
        let km3_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[4144..5168],
            64u32,
        );
        let vb23_4 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_4, &vb23_4, &s4, &s4);
        let km3_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[5168..6192],
            64u32,
        );
        let vb23_5 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_5, &vb23_5, &s5, &s5);
        let km3_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[6192..7216],
            64u32,
        );
        let vb23_6 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_6, &vb23_6, &s6, &s6);
        let km3_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &kgd_sh[7216..8240],
            64u32,
        );
        let vb23_7 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &vn16_sh[768..1024],
            16u32,
        );
        cmma::execute(&km3_7, &vb23_7, &s7, &s7);
    }
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
use burn::tensor::{Device, Distribution, Tensor};
use burn_gdn2::kernel::chunk_cube::cuda::{inter_launch_raw, intra_launch_raw};
use burn_gdn2::Gdn2Config;

#[test]
#[ignore]
fn rr_inter_bench() {
    const HS: usize = 1024;
    const HEADS: usize = 8;
    const HK: usize = 128;
    const CS: usize = 64;
    const SEQ: usize = 4096;
    const NT: usize = SEQ / CS;
    const VD: usize = HK;
    const C: usize = CS;
    const KD: usize = HK;
    type Bare = burn_gdn2::CudaBare;
    let dev = Device::cuda(0);
    let cfg = Gdn2Config {
        hidden_size: HS,
        num_heads: HEADS,
        head_dim: HK,
        mode: burn_gdn2::Gdn2Mode::Chunk,
        ..Default::default()
    };
    let m = burn_gdn2::GatedDeltaNet2::new(&cfg, &dev);
    let x = Tensor::<3>::random([1, SEQ, HS], Distribution::Normal(0.0, 1.0), &dev);
    let (proj, _) = m.project(x.clone(), None);
    let state = Tensor::<4>::zeros([1, HEADS, KD, VD], &dev);
    let scale = (HK as f64).powf(-0.5);
    let io = intra_launch_raw::<Bare>(
        proj.q,
        proj.k,
        proj.v,
        proj.g,
        proj.b,
        proj.w,
        state.clone(),
        scale,
        CS,
        31,
        8,
    );
    let _ = io.out.clone().into_data();
    // старый inter: 5 вызовов + 1 sync
    for _ in 0..3 {
        let o = inter_launch_raw::<Bare>(
            io.aqk.clone(),
            io.w.clone(),
            io.u.clone(),
            io.qgt.clone(),
            io.kgd.clone(),
            io.glast.clone(),
            state.clone(),
            io.out.clone(),
            scale,
            CS,
            NT,
            HEADS,
            2,
            1,
            7,
        );
        let _ = o.clone().into_data();
    }
    let t0 = std::time::Instant::now();
    let mut last = None;
    for _ in 0..5 {
        last = Some(inter_launch_raw::<Bare>(
            io.aqk.clone(),
            io.w.clone(),
            io.u.clone(),
            io.qgt.clone(),
            io.kgd.clone(),
            io.glast.clone(),
            state.clone(),
            io.out.clone(),
            scale,
            CS,
            NT,
            HEADS,
            2,
            1,
            7,
        ));
    }
    let lc = last.unwrap().try_into_primitive::<Bare>().unwrap();
    let _ = lc.client.read_one_unchecked(lc.handle);
    println!("old inter xl: {:?}us", t0.elapsed().as_micros() / 5);
    // rr inter
    let f = |t: &Tensor<3>| t.clone().try_into_primitive::<Bare>().unwrap();
    let g = |t: &Tensor<2>| t.clone().try_into_primitive::<Bare>().unwrap();
    let st = state.clone().try_into_primitive::<Bare>().unwrap();
    let w_c = f(&io.w);
    let u_c = f(&io.u);
    let qg_c = f(&io.qgt);
    let kgd_c = f(&io.kgd);
    let aqk_c = f(&io.aqk);
    let glast_c = g(&io.glast);
    let client = st.client.clone();
    let out_h = client.empty(HEADS * NT * C * VD * 4);
    let state_out_h = client.empty(HEADS * KD * VD * 4);
    for _ in 0..3 {
        unsafe {
            rr_inter_kernel::launch(
                &client,
                CubeCount::Static(HEADS as u32, (VD / 16) as u32, 1),
                CubeDim { x: 32, y: 1, z: 1 },
                BufferArg::from_raw_parts(w_c.handle.clone(), HEADS * NT * KD * C),
                BufferArg::from_raw_parts(u_c.handle.clone(), HEADS * NT * C * VD),
                BufferArg::from_raw_parts(qg_c.handle.clone(), HEADS * NT * KD * C),
                BufferArg::from_raw_parts(kgd_c.handle.clone(), HEADS * NT * C * KD),
                BufferArg::from_raw_parts(aqk_c.handle.clone(), HEADS * NT * C * C),
                BufferArg::from_raw_parts(glast_c.handle.clone(), HEADS * NT * KD),
                BufferArg::from_raw_parts(st.handle.clone(), HEADS * KD * VD),
                BufferArg::from_raw_parts(state_out_h.clone(), HEADS * KD * VD),
                BufferArg::from_raw_parts(out_h.clone(), HEADS * NT * C * VD),
                NT as u32,
                scale as f32,
            );
        }
        let _ = client.read_one_unchecked(out_h.clone());
    }
    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        unsafe {
            rr_inter_kernel::launch(
                &client,
                CubeCount::Static(HEADS as u32, (VD / 16) as u32, 1),
                CubeDim { x: 32, y: 1, z: 1 },
                BufferArg::from_raw_parts(w_c.handle.clone(), HEADS * NT * KD * C),
                BufferArg::from_raw_parts(u_c.handle.clone(), HEADS * NT * C * VD),
                BufferArg::from_raw_parts(qg_c.handle.clone(), HEADS * NT * KD * C),
                BufferArg::from_raw_parts(kgd_c.handle.clone(), HEADS * NT * C * KD),
                BufferArg::from_raw_parts(aqk_c.handle.clone(), HEADS * NT * C * C),
                BufferArg::from_raw_parts(glast_c.handle.clone(), HEADS * NT * KD),
                BufferArg::from_raw_parts(st.handle.clone(), HEADS * KD * VD),
                BufferArg::from_raw_parts(state_out_h.clone(), HEADS * KD * VD),
                BufferArg::from_raw_parts(out_h.clone(), HEADS * NT * C * VD),
                NT as u32,
                scale as f32,
            );
        }
    }
    let _ = client.read_one_unchecked(out_h.clone());
    println!("rr inter xl: {:?}us", t0.elapsed().as_micros() / 5);
}
