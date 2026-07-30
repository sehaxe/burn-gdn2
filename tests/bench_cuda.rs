#![allow(missing_docs, non_snake_case, dead_code, unused_imports, unused_variables)]
//! burn-gdn2 CUDA benchmark — projections + chunk forward.
//! Run: cargo test --release --features cuda -p burn-gdn2 --test bench_cuda -- --ignored --nocapture

#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use burn::tensor::{Distribution, Tensor};
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
#[cfg(feature = "cuda")]
use burn::module::Module;
#[cfg(feature = "cuda")]
use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2};
#[cfg(feature = "cuda")]
use burn_gdn2::forward::chunk_wy_forward;

#[cfg(feature = "cuda")]
type B = Cuda;

#[cfg(feature = "cuda")]
fn time_it(runs: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..(runs.min(5)) { f(); }
    let start = Instant::now();
    for _ in 0..runs { f(); }
    start.elapsed().as_secs_f64() / runs as f64
}

#[cfg(feature = "cuda")]
fn cfg(hs: usize, nh: usize, hk: usize, ev: f32, nvh: Option<usize>, cs: usize) -> Gdn2Config {
    Gdn2Config {
        hidden_size: hs, num_heads: nh, head_dim: hk, expand_v: ev,
        num_v_heads: nvh, chunk_size: cs, mode: Gdn2Mode::Chunk,
        ..Default::default()
    }
}

#[cfg(feature = "cuda")]
const CFGS: &[(&str, usize, usize, usize, f32, Option<usize>, usize, usize, usize, usize)] = &[
    ("tiny",    64,  2, 32, 1.0, None,      64, 1,   1,  64),
    ("small",  128,  4, 32, 1.0, None,      64, 1,   1, 256),
    ("med",    256,  4, 64, 1.0, None,      64, 1, 128, 1024),
    ("med_f",  256,  4, 64, 1.0, None,      64, 1, 256, 1024),
    ("large",  512,  8, 64, 1.0, None,      64, 1, 256, 2048),
    ("xl",    1024,  8,128, 1.0, None,      64, 1, 512, 4096),
    ("gva_4x", 512,  4, 64, 1.0, Some(16),  64, 1, 256, 2048),
    ("exp_v2", 512,  8, 64, 2.0, None,      64, 1, 256, 2048),
];

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_cuda() {
    let dev = Default::default();
    println!("\n{:=^80}", "");
    println!("  burn-gdn2 CUDA BENCHMARK — RTX 5060 Ti");
    println!("{:=^80}", "");

    println!("\n── 1. PROJECTIONS ONLY ──");
    println!("{:<12} {:>10} {:>10}", "config", "project_us", "B*Sq");

    for &(name, hs, nh, hk, ev, nvh, _cs, B, Sq, Sk) in CFGS {
        let c = cfg(hs, nh, hk, ev, nvh, 64);
        let m = GatedDeltaNet2::<B>::new(&c, &dev);
        let x = Tensor::<B, 3>::random([B, Sk, hs], Distribution::Normal(0.0, 1.0), &dev);

        let proj_time = time_it(50, {
            let m = m.clone(); let x = x.clone();
            move || { let _ = m.project(x.clone()); }
        });

        println!("{:<12} {:>10.0} {:>10}", name, proj_time * 1e6, B * Sk);
    }

    println!("\n── 2. CHUNK FORWARD ──");
    println!("{:<12} {:>10} {:>10} {:>6}", "config", "chunk_us", "seq", "tok/s");
    for &(name, hs, nh, hk, ev, nvh, cs, B, Sq, Sk) in CFGS {
        if Sk < 128 { continue; }
        let c = cfg(hs, nh, hk, ev, nvh, cs);
        let m = GatedDeltaNet2::<B>::new(&c, &dev);
        let x = Tensor::<B, 3>::random([B, Sk, hs], Distribution::Normal(0.0, 1.0), &dev);

        let runs = if Sk <= 1024 { 20 } else { 5 };
        let t = time_it(runs, {
            let m = m.clone(); let x = x.clone();
            move || { let _ = m.forward_train(x.clone()); }
        });

        let tok_s = (B * Sk) as f64 / t;
        println!("{:<12} {:>10.0} {:>10} {:>8.0}", name, t * 1e6, Sk, tok_s);
    }

    println!();
}
