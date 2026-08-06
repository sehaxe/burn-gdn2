//! Stable chunking: chunk_wy_forward must match the exact per-token
//! recurrence (kernel::fused_recurrent_forward) for any chunk size,
//! including weak decay (g ~ -0.05) where the naive exp(cumsum(g))
//! factor overflows f32 at chunks > ~17.

use burn::backend::{ndarray::NdArrayDevice, NdArray};
use burn::tensor::{Distribution, Tensor};
use burn_gdn2::chunk_wy_forward;
use burn_gdn2::kernel::fused_recurrent::fused_recurrent_forward;

fn run(decay_lo: f32, decay_hi: f32) {
    let device = NdArrayDevice::Cpu;
    let (b, h, t, k, v) = (2, 3, 96, 8, 8);

    let q = Tensor::<NdArray, 4>::random([b, h, t, k], Distribution::Normal(0.0, 1.0), &device);
    let kk_raw = Tensor::<NdArray, 4>::random([b, h, t, k], Distribution::Normal(0.0, 1.0), &device);
    let kk = kk_raw.clone() / (kk_raw.clone() * kk_raw.clone()).sum_dim(3).sqrt();
    let vv = Tensor::<NdArray, 4>::random([b, h, t, v], Distribution::Normal(0.0, 1.0), &device);
    let g = Tensor::<NdArray, 4>::random(
        [b, h, t, 1],
        Distribution::Uniform(decay_lo as f64, decay_hi as f64),
        &device,
    );
    let bb = Tensor::<NdArray, 4>::random([b, h, t, 1], Distribution::Uniform(0.0, 1.0), &device);
    let w = Tensor::<NdArray, 4>::random([b, h, t, v], Distribution::Normal(0.0, 1.0), &device);

    let state0 = Tensor::zeros([b, h, k, v], &device);
    let (out_ref, state_ref) = fused_recurrent_forward(
        q.clone(),
        kk.clone(),
        vv.clone(),
        g.clone(),
        bb.clone(),
        w.clone(),
        state0.clone(),
        1.0,
    );

    for chunk_size in [16usize, 64] {
        let (out, state) = chunk_wy_forward(
            q.clone(),
            kk.clone(),
            vv.clone(),
            g.clone(),
            bb.clone(),
            w.clone(),
            state0.clone(),
            1.0,
            chunk_size,
        );

        let out_diff = (out.clone() - out_ref.clone()).abs().max().to_data().to_vec::<f32>().unwrap()[0];
        let state_diff = (state.clone() - state_ref.clone()).abs().max().to_data().to_vec::<f32>().unwrap()[0];
        assert!(
            out_diff < 1e-3,
            "chunk {chunk_size}: out max diff {out_diff} (decay {decay_lo}..{decay_hi})"
        );
        assert!(
            state_diff < 1e-3,
            "chunk {chunk_size}: state max diff {state_diff} (decay {decay_lo}..{decay_hi})"
        );
    }
}

#[test]
fn stable_weak_decay() {
    run(-0.3, -0.02);
}

#[test]
fn stable_strong_decay() {
    run(-5.0, -0.5);
}
