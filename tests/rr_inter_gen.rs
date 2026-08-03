//! Register-resident inter recurrence with the REAL GDN-2 step (matches
//! `gdn2_chunk_inter_kernel` math): vn = u - w@S, out = aqk@vn + scale*qg@S,
//! S = diag(glast)@S + kgd^T@vn. All dots via tensor cores (f16, f32 acc).
//! One warp per (head, v-slice of 16); S in 4 accumulators.

use cubecl::cmma;
use cubecl::prelude::*;
use half::f16;

const KD: usize = 64;
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
    let u_stride = C * 64;
    let out_stride = C * 64;
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
        s32_sh[i] = state_in[base * KD * 64 + (i / 16) * 64 + n0 + (i % 16)];
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
            u_sh[t * 16 + v] = f16::cast_from(u_all[hw * u_stride + t * 64 + n0 + v]);
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
            &w_sh[0..1024],
            64u32,
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
            &w_sh[1024..2048],
            64u32,
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
            &w_sh[2048..3072],
            64u32,
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
            &w_sh[3072..4096],
            64u32,
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
            &w_sh[16..1040],
            64u32,
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
            &w_sh[1040..2064],
            64u32,
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
            &w_sh[2064..3088],
            64u32,
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
            &w_sh[3088..4112],
            64u32,
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
            &w_sh[32..1056],
            64u32,
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
            &w_sh[1056..2080],
            64u32,
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
            &w_sh[2080..3104],
            64u32,
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
            &w_sh[3104..4128],
            64u32,
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
            &w_sh[48..1072],
            64u32,
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
            &w_sh[1072..2096],
            64u32,
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
            &w_sh[2096..3120],
            64u32,
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
            &w_sh[3120..4144],
            64u32,
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
            &qg_sh[0..1024],
            64u32,
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
            &qg_sh[1024..2048],
            64u32,
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
            &qg_sh[2048..3072],
            64u32,
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
            &qg_sh[3072..4096],
            64u32,
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
            &qg_sh[16..1040],
            64u32,
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
            &qg_sh[1040..2064],
            64u32,
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
            &qg_sh[2064..3088],
            64u32,
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
            &qg_sh[3088..4112],
            64u32,
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
            &qg_sh[32..1056],
            64u32,
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
            &qg_sh[1056..2080],
            64u32,
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
            &qg_sh[2080..3104],
            64u32,
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
            &qg_sh[3104..4128],
            64u32,
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
            &qg_sh[48..1072],
            64u32,
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
            &qg_sh[1072..2096],
            64u32,
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
            &qg_sh[2096..3120],
            64u32,
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
            &qg_sh[3120..4144],
            64u32,
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
            out[hw * out_stride + (i / 16) * 64 + n0 + (i % 16)] = o32_sh[i];
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
    }
    let sb = state_out.slice_mut(base * KD * 64, (base + 1) * KD * 64);
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
    for i in 0..KD * 16 {
        sb[(i / 16) * 64 + n0 + (i % 16)] = s32_sh[i];
    }
}
