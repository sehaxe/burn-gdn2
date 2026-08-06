#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! Fused chunk kernels vs tensor path: numerical equivalence on CUDA.
//! Run: cargo test --release --features "cuda" --test fused_chunk_verify -- --nocapture

use burn::backend::{Backend, NdArray};
use burn::tensor::{Device, Distribution, Tensor};
use burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward;
use burn_gdn2::{chunk_wy_forward, CudaBare};

fn max_rel(a: &Tensor<4>, b: &Tensor<4>) -> f32 {
    let a = a.clone().into_data();
    let b = b.clone().into_data();
    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    for (x, y) in a.bytes.chunks_exact(4).zip(b.bytes.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        max_abs = max_abs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    max_abs / scale.max(1e-30)
}

#[test]
fn fused_chunk_matches_tensor_path() {
    type B = CudaBare;
    let dev: Device = Default::default();
    // chunk <= 16: the fused kernels' numerical range (K3 design). Larger
    // chunks fall back to the 16-tile tensor path (checked separately).
    for (batch, heads, time, k_dim, v_dim, cs) in [
        (1usize, 4usize, 64usize, 64usize, 64usize, 16usize),
        (1usize, 4usize, 256usize, 64usize, 64usize, 16usize),
        (2, 8, 2048, 64, 64, 16),
        (1, 8, 4096, 128, 128, 16),
    ] {
        let q = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let k = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let v = Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let g = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(-0.05, 0.1),
            &dev,
        );
        let b = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Uniform(0.0, 0.1),
            &dev,
        );
        let w = Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Uniform(0.0, 0.1),
            &dev,
        );
        let s = Tensor::<4>::random(
            [batch, heads, k_dim, v_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let scale = (k_dim as f64).powf(-0.5);

        let (ref_out, ref_s) = chunk_wy_forward(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            s.clone(),
            scale,
            cs,
        );
        let (f_out, f_s) = fused_chunk_forward::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            s.clone(),
            scale,
            cs,
        )
        .expect("fused path should dispatch");

        let do_out = max_rel(&f_out, &ref_out);
        let d_s = max_rel(&f_s, &ref_s);
        println!(
            "B={batch} H={heads} T={time} k={k_dim} v={v_dim}: out_rel={do_out:.2e} state_rel={d_s:.2e}"
        );
        assert!(do_out < 1e-3, "out mismatch: {do_out:.2e}");
        assert!(d_s < 1e-3, "state mismatch: {d_s:.2e}");
    }
    let _ = NdArray::name(&Default::default());
}

/// The fused-op backward with the kernel-exported M^-1 must match the
/// tensor-path gradients on CUDA (within fp32 noise from the kernel).
#[test]
fn fused_op_grads_match_tensor_path_cuda() {
    use burn::tensor::Device as D;
    let plain: D = Default::default();
    let dev = burn::tensor::Device::autodiff(plain);
    let (batch, heads, time, k_dim, v_dim, cs) =
        (1usize, 2usize, 128usize, 32usize, 32usize, 16usize);
    let scale = (k_dim as f64).powf(-0.5);
    let mk = |shape: [usize; 4], dist: Distribution| {
        Tensor::<4>::random(shape, dist, &dev).require_grad()
    };
    let q = mk([batch, heads, time, k_dim], Distribution::Normal(0.0, 0.1));
    let k = mk([batch, heads, time, k_dim], Distribution::Normal(0.0, 0.1));
    let v = mk([batch, heads, time, v_dim], Distribution::Normal(0.0, 0.1));
    let g = mk([batch, heads, time, k_dim], Distribution::Normal(-0.5, 0.2));
    let b = mk([batch, heads, time, k_dim], Distribution::Uniform(0.0, 0.1));
    let w = mk([batch, heads, time, v_dim], Distribution::Uniform(0.0, 0.1));
    let s = mk([batch, heads, k_dim, v_dim], Distribution::Normal(0.0, 0.1));

    let (out, _) = chunk_wy_forward(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b.clone(),
        w.clone(),
        s.clone(),
        scale,
        cs,
    );
    let grads_ref = out.powf_scalar(2.0).sum().backward();
    let (out2, _) = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b.clone(),
        w.clone(),
        s.clone(),
        scale,
        cs,
    )
    .expect("op should dispatch");
    let grads_f = out2.powf_scalar(2.0).sum().backward();

    for (name, t) in [
        ("q", &q),
        ("k", &k),
        ("v", &v),
        ("g", &g),
        ("b", &b),
        ("w", &w),
        ("s", &s),
    ] {
        let gr = t.grad(&grads_ref).unwrap().clone();
        let gf = t.grad(&grads_f).unwrap().clone();
        let a = gr.clone().into_data();
        let b = gf.clone().into_data();
        let mut max_abs = 0.0f32;
        let mut scale_v = 0.0f32;
        for (x, y) in a.bytes.chunks_exact(4).zip(b.bytes.chunks_exact(4)) {
            let x = f32::from_le_bytes(x.try_into().unwrap());
            let y = f32::from_le_bytes(y.try_into().unwrap());
            max_abs = max_abs.max((x - y).abs());
            scale_v = scale_v.max(x.abs()).max(y.abs());
        }
        let rel = max_abs / scale_v.max(1e-30);
        assert!(rel < 1e-2, "{name}: grads mismatch rel={rel:.2e}");
        println!("{name}: grads rel={rel:.2e}");
    }
}
