#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! Training + forward benchmark: fused autodiff chunk op vs plain tensor path
//! vs PyTorch (see `bench_torch.py`). Same configs as the torch side.
//! Run: cargo test --release --features "cuda,autodiff" --test bench_train_cuda -- --ignored --nocapture

use burn::backend::Autodiff;
use burn::tensor::{Distribution, Tensor};
use burn_gdn2::{CudaBare, GatedDeltaNet2, Gdn2Config, Gdn2Mode};

type AD = Autodiff<CudaBare>;

fn time_it(runs: usize, mut f: impl FnMut()) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        f();
    }
    t0.elapsed().as_secs_f64() / runs as f64
}

#[test]
#[ignore]
fn bench_train_cuda() {
    let cfg_of = |d: usize, h: usize, hk: usize| Gdn2Config {
        hidden_size: d,
        num_heads: h,
        head_dim: hk,
        mode: Gdn2Mode::Chunk,
        chunk_size: 64,
        ..Default::default()
    };

    println!("── CUDA: burn-gdn2 (forward / train fwd+bwd) vs torch (chunked WY) ──");
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "config", "b_fwd_ms", "b_train_ms", "t_fwd_ms", "t_train_ms", "fwd_x", "train_x", "tok/s"
    );

    for (d, h, hk, t) in [
        (256usize, 4usize, 64usize, 256usize),
        (512, 8, 64, 1024),
        (1024, 8, 128, 2048),
        (2048, 16, 128, 4096),
    ] {
        let cfg = cfg_of(d, h, hk);
        let plain: burn::tensor::Device = Default::default();
        let ad_dev = burn::tensor::Device::autodiff(plain.clone());

        let m_plain = GatedDeltaNet2::new(&cfg, &plain);
        let m_ad = GatedDeltaNet2::new(&cfg, &ad_dev);
        let x_plain = Tensor::<3>::random([1, t, d], Distribution::Normal(0.0, 1.0), &plain);
        let x_ad = Tensor::<3>::random([1, t, d], Distribution::Normal(0.0, 1.0), &ad_dev);

        let _ = m_plain
            .forward_train::<CudaBare>(x_plain.clone())
            .into_data();
        let _ = {
            let loss = m_ad
                .forward_train_fused::<AD>(x_ad.clone())
                .powf_scalar(2.0)
                .mean();
            let _ = loss.clone().into_data();
            loss.backward()
        };
        let _ = {
            let loss = m_ad
                .forward_train::<AD>(x_ad.clone())
                .powf_scalar(2.0)
                .mean();
            let _ = loss.clone().into_data();
            loss.backward()
        };
        let runs = if t <= 1024 { 5 } else { 3 };
        let t_fwd = time_it(runs, || {
            let _ = m_plain
                .forward_train::<CudaBare>(x_plain.clone())
                .into_data();
        });
        let _ = m_ad.forward_train_fused::<AD>(x_ad.clone());
        let t_train_fused = time_it(runs, || {
            let loss = m_ad
                .forward_train_fused::<AD>(x_ad.clone())
                .powf_scalar(2.0)
                .mean();
            let _ = loss.clone().into_data();
            let _grads = loss.backward();
        });
        let t_train_plain = time_it(runs, || {
            let loss = m_ad
                .forward_train::<AD>(x_ad.clone())
                .powf_scalar(2.0)
                .mean();
            let _ = loss.clone().into_data();
            let _grads = loss.backward();
        });

        // torch timings (bench_torch.py, same configs, same GPU)
        let (t_fwd_t, t_train_t) = match (d, h, hk, t) {
            (256, 4, 64, 256) => (52.1e-3, 260.3e-3),
            (512, 8, 64, 1024) => (195.6e-3, 1140.6e-3),
            (1024, 8, 128, 2048) => (402.9e-3, 2188.5e-3),
            (2048, 16, 128, 4096) => (804.7e-3, 4236.8e-3),
            _ => unreachable!(),
        };
        println!(
            "{:<16} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>7.1}x {:>7.1}x {:>6.0}",
            format!("d={d}, T={t}"),
            t_fwd * 1e3,
            t_train_fused * 1e3,
            t_fwd_t * 1e3,
            t_train_t * 1e3,
            t_fwd_t / t_fwd,
            t_train_t / t_train_fused,
            t as f64 / t_train_fused,
        );
        let _ = t_train_plain;
    }
}
