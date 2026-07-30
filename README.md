# burn-gdn2 - Gated DeltaNet 2

[![Crates.io](https://img.shields.io/crates/v/burn-gdn2)](https://crates.io/crates/burn-gdn2)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)
[![MS-DSA](https://img.shields.io/badge/stack-Burn%20%2B%20cubecl-purple.svg)](#architecture)

**Linear-complexity recurrent token mixer** for [Burn](https://burn.dev).
Channel-wise erase and write gates replace the scalar write-strength gate of
the gated delta rule. Two forward modes: token-by-token recurrent (inference)
and chunked WY decomposition (training).

O(T) complexity instead of O(T²) - trains on 4K+ sequences on a single GPU.

## Requirements

- **Rust**: ≥ 1.75
- **Burn**: 0.21
- **GPU** (optional): CUDA or Vulkan/Metal via WGPU

## Install

```bash
cargo add burn-gdn2
```

Enable CUDA:

```toml
burn-gdn2 = { version = "0.1", features = ["cuda"] }
```

## Quick start

```rust
use burn_ndarray::NdArray;
use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2};

let device = Default::default();
let cfg = Gdn2Config {
    hidden_size: 256,
    num_heads: 4,
    head_dim: 64,
    mode: Gdn2Mode::Chunk,
    ..Default::default()
};
let model = GatedDeltaNet2::<NdArray>::new(&cfg, &device);

// Training - chunked WY forward
let x = Tensor::random([1, 1024, 256], Distribution::Normal(0.0, 1.0), &device);
let output = model.forward_train(x);

// Inference - token-by-token with persistent state
let mut state = None;
let token = Tensor::random([1, 1, 256], Distribution::Normal(0.0, 1.0), &device);
let output = model.forward(token, &mut state, true);
```

## Architecture

```
x ──→ Q/K/V/B/W/G projections ──→ ShortConv ──→ L2-Norm
                                       │
                    ┌──────────────────┤
                    ▼                  ▼
            Chunk WY (train)    Fused Recurrent (infer)
                    │                  │
                    ▼                  ▼
              Gated RMS Norm ←── SiLU Gate
                    │
                    ▼
               Output Proj
```

Per-token recurrence on matrix state S ∈ R^{d_k × d_v}:

```
S_t = (I - k_t (b_t ⊙ k_t)^T) diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)^T
o_t = q_t^T S
```

Where `b_t` is the channel-wise erase gate (key axis) and `w_t` is the
channel-wise write gate (value axis), replacing the scalar gate of the gated delta rule.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `hidden_size` | 2048 | Model dimension |
| `num_heads` | 16 | Query/key heads |
| `head_dim` | 128 | Dimension per key/query head |
| `expand_v` | 1.0 | Value dimension expansion factor |
| `num_v_heads` | None | Value heads (None = same as num_heads) |
| `use_short_conv` | true | Depthwise conv before recurrence |
| `allow_neg_eigval` | false | Allow negative eigenvalues (b × 2) |
| `mode` | FusedRecurrent | Chunk (train) or FusedRecurrent (infer) |
| `chunk_size` | 64 | Chunk size for Chunk mode |
| `norm_eps` | 1e-5 | Output norm epsilon |

## Performance

Benchmarked on RTX 5060 Ti, Burn 0.21 vs equivalent PyTorch chunk implementation.

**Chunk forward** (training mode) is **2.2–5.5× faster** across all configs -
batched matmul for cumulative sum and pre-allocated slice_assign replace
O(T) CPU-side loops.

**Projections only** (Q/K/V/B/W/G) are 0.6–0.8× vs PyTorch - Burn's cubecl
matmul overhead for small matrix sizes. This is a fixed cost per layer and
does not scale with sequence length.

| Config | Burn CUDA | PyTorch CUDA | Speedup |
|--------|-----------|--------------|---------|
| seq=256, d=128 | 29 ms | 63 ms | **2.2×** |
| seq=1024, d=256 | 117 ms | 267 ms | **2.3×** |
| seq=2048, d=512 | 221 ms | 645 ms | **2.9×** |
| seq=4096, d=1024 | 341 ms | 1,875 ms | **5.5×** |

## License

AGPL-3.0. See [LICENSE](LICENSE).
