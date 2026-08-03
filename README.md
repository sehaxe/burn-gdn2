# burn-gdn2 - Gated DeltaNet 2

[![Crates.io](https://img.shields.io/crates/v/burn-gdn2)](https://crates.io/crates/burn-gdn2)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)
[![docs.rs](https://img.shields.io/docsrs/burn-gdn2)](https://docs.rs/burn-gdn2)

**Linear-complexity recurrent token mixer** for [Burn](https://burn.dev),
implementing Gated DeltaNet 2 ([NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2)).
Channel-wise erase and write gates replace the scalar write-strength gate of
the gated delta rule. O(T) complexity instead of O(T²).

## Why this crate instead of the reference repo

| | burn-gdn2 | NVlabs reference |
|---|---|---|
| Backends | CPU, CUDA, Vulkan, Metal, WGPU (any burn backend) | **NVIDIA only** (Triton + flash-attn) |
| Dependencies | burn only | lit-gpt + fla + einops + transformers + triton + flash-attn |
| Backward pass | automatic (burn autodiff) | hand-written Triton kernels per op |
| Verification | 1000-case bit-exact reference tests (regenerated with `tests/gen_reference.py`) | none shipped |
| Extras | `min_decay` (per-channel decay floor), GVA, `allow_neg_eigval` | GVA, `allow_neg_eigval` |

## Install

```bash
cargo add burn-gdn2
```

Enable CUDA:

```toml
burn-gdn2 = { version = "0.5", features = ["cuda"] }
```

## Quick start

```rust
use burn::tensor::{backend::Backend, Distribution, Tensor};
use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2, Gdn2State};

type B = burn_ndarray::NdArray;
let device = B::Device::default();
let cfg = Gdn2Config {
    hidden_size: 256,
    num_heads: 4,
    head_dim: 64,
    mode: Gdn2Mode::Chunk, // training mode
    ..Default::default()
};
let model = GatedDeltaNet2::<B>::new(&cfg, &device);

// Training - chunked WY forward.
let x = Tensor::<B, 3>::random([1, 1024, 256], Distribution::Normal(0.0, 1.0), &device);
let output = model.forward_train(x);

// Inference - prefill, then decode token-by-token with persistent state.
let mut state: Option<Gdn2State<B>> = None;
let out = model.forward(x, &mut state, true);          // prefill (or any chunk)
let token = Tensor::<B, 3>::random([1, 1, 256], Distribution::Normal(0.0, 1.0), &device);
let next = model.forward(token, &mut state, true);     // decode
```

`Gdn2State` carries both the recurrent matrix state `S [B, HV, K, V]` and the
short-convolution caches. This is what makes token-by-token decoding **exactly
equivalent** to one forward pass over the full sequence (verified bit-exact by
`tests/autodiff.rs`).

## Architecture

```
x ──→ Q/K/V/B/W/G projections ──→ ShortConv ──→ per-head L2-Norm
                                        │
                     ┌──────────────────┤
                     ▼                  ▼
             Chunk WY (train/long)  Fused Recurrent (decode)
                     │                  │
                     ▼                  ▼
               Gated RMS Norm ←── SiLU Gate
                     │
                     ▼
                Output Proj
```

Per-token recurrence on the per-head matrix state `S ∈ R^{d_k × d_v}`:

```
S_t = (I - k_t (b_t ⊙ k_t)^T) diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)^T
o_t = q_t^T S_t
```

`b_t` is the channel-wise erase gate (key axis), `w_t` the channel-wise write
gate (value axis); `α_t` is the per-channel decay from `A_log` and
`softplus(F(x) + dt_bias)`.

### Fused CUDA kernels (v0.5+)

On the bare CUDA backend the whole chunked forward runs as **two custom cubecl
kernels** instead of the per-chunk tensor-op loop:

- `gdn2_chunk_intra_kernel` — chunk-local precomputation (decays, score
  matrices, `A = (I + T)^{-1}`, pseudo-keys/values) in a single launch.
- `gdn2_chunk_inter_kernel` — the sequential recurrence across chunks with
  the in-loop output phase, prefetching `w`/`kgd` through shared memory when
  the head dim allows.

Dispatch is gated on `burn_gdn2::CudaBare` (the bare `CubeBackend`, exported
for convenience); every other backend transparently falls back to the tensor
path. The fused path is numerically verified against the tensor path
(`fused_kernel_matches_tensor_path` in `tests/bench_cuda.rs`).

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `hidden_size` | 2048 | Model dimension |
| `num_heads` | 16 | Query/key heads |
| `head_dim` | 128 | Dimension per key/query head |
| `expand_v` | 1.0 | Value expansion (must keep `head_v_dim` integer) |
| `num_v_heads` | None | Value heads; GVA when > `num_heads` (must divide it) |
| `use_short_conv` | true | Causal depthwise conv (kernel 4) before the recurrence |
| `allow_neg_eigval` | false | Lift erase gate to `[0, 2]` (negative eigenvalues) |
| `mode` | FusedRecurrent | `Chunk` (training) or `FusedRecurrent` |
| `chunk_size` | 64 | Chunk size for `Chunk` mode |
| `norm_eps` | 1e-5 | Output norm epsilon |
| `min_decay` | None | Optional per-channel decay floor (extension, not in the paper) |

Invalid configurations (fractional `expand_v`, non-divisible `num_v_heads`,
`num_v_heads < num_heads`, `chunk_size = 0`) panic with a clear message at
construction.

## Performance

Measured on RTX 5060 Ti, release build, Burn 0.21, fp32, best-of-N loop
timing. The fused path (and therefore the comparison) is **NVIDIA/CUDA only**
— the fused kernels are gated on the bare CUDA backend; every other backend
falls back to the tensor path. `burn-gdn2` runs through the fused chunked CUDA
kernels; the NVlabs side is their Triton implementation (`chunk_gdn2`).

| Config | burn-gdn2 | NVlabs Triton | vs NVlabs |
|--------|-----------|---------------|-----------|
| d=256, T=256 | 0.076 ms | 0.561 ms | **7.4×** |
| d=512, T=1024 | 0.237 ms | 0.536 ms | **2.3×** |
| d=1024, T=2048 | 0.868 ms | 0.628 ms | 1.4× slower |
| d=2048, T=4096 | 3.887 ms | 1.461 ms | 2.7× slower |

Small/medium configs win by a wide margin; long sequences with wide heads
(large/xl) still lose to the Triton recurrence, whose register-resident state
avoids our per-chunk shared-memory barrier chain. The chunk-size is fixed at
64 in both implementations.

### Per-operation breakdown (d=2048, T=4096, xl)

The reference implementation splits the forward into separate Triton kernels;
the equivalent work inside our two fused kernels breaks down as follows
(isolated phase timings; the fused total is less than the sum because the
phases overlap across blocks):

| Operation | burn-gdn2 | NVlabs Triton | vs NVlabs |
|-----------|-----------|---------------|-----------|
| chunk-local precompute + solve (intra) | 1.86 ms | 0.78 ms | 2.4× slower |
| recurrence (v_new + state update) | 1.40 ms | 0.28 ms | 5.0× slower |
| output (aqk·v_new + qg·S) | 1.29 ms | 0.30 ms | 4.3× slower |
| decay cumsum | fused into intra | 0.08 ms | — |

The recurrence and output phases are where the Triton kernels pull ahead:
register-resident state with tensor-core dots, no per-chunk barriers. These
two phases are the target of ongoing kernel work.

Run it yourself:

```bash
# burn side (fused path, correctness-verified against the tensor path)
cargo test --release --features cuda -p burn-gdn2 --test bench_cuda -- --ignored --nocapture
# NVlabs side (same GPU, same configs)
python3 bench_torch.py
```

Notes:
- `Chunk` mode wins for training and long sequences; `forward()` dispatches
  to fused automatically for sequences of ≤ 64 tokens (decode).
- Projection costs are identical to any linear layer; the recurrence itself
  is the constant part.
- Per-token decode (1 token per call, ~1.2k tok/s on the RTX 5060 Ti) is
  dominated by the projections; the recurrence step itself runs through a
  single fused cubecl kernel (`src/kernel/fused_recurrent_cube.rs`) instead
  of ~8 tensor ops. Dispatch is gated on the bare CUDA backend
  (`burn_gdn2::CudaBare`, re-exported for convenience); the default
  `burn_cuda::Cuda` is `Fusion<CubeBackend>`-wrapped and has no public way to
  hand out its underlying `CubeTensor`, so it transparently falls back to the
  tensor path.

## Tests

```bash
cargo test -p burn-gdn2                         # unit + autodiff + decode tests
cargo test -p burn-gdn2 --features binary-tests # 1000-case bit-exact vs torch reference
cargo test -p burn-gdn2 --features cuda         # CUDA backend
python3 tests/gen_reference.py                  # regenerate tests/ref_data.bin
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
