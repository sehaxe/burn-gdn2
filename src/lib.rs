//! # burn-gdn2
//!
//! Gated DeltaNet 2 (GDN-2) — a linear‑complexity recurrent token mixer
//! with channel‑wise erase/write gates.
//!
//! ## Quick start
//!
//! ```rust
//! use burn::backend::NdArray;
//! use burn::tensor::Tensor;
//! use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2};
//!
//! let device = burn::backend::ndarray::NdArrayDevice::Cpu;
//! let config = Gdn2Config {
//!     hidden_size: 64,
//!     num_heads: 2,
//!     head_dim: 32,
//!     ..Default::default()
//! };
//! let model = GatedDeltaNet2::<NdArray>::new(&config, &device);
//!
//! // Inference — token‑by‑token, state passed by reference
//! let input = Tensor::zeros([1, 16, 64], &device);
//! let mut state = None;
//! let output = model.forward(input, &mut state, true);
//!
//! // Training — full sequence, chunked WY for efficiency
//! let input = Tensor::zeros([1, 128, 64], &device);
//! let output = model.forward_train(input);
//! ```
//!
//! ## Features
//!
//! - **`std`** (default) — standard library support
//! - **`autodiff`** — differentiation support (required for training)
//! - **`cubecl`** — CubeCL‑accelerated GPU kernels (experimental)

pub mod config;
pub mod forward;
pub mod kernel;
pub mod l2norm;
pub mod module;
pub mod short_conv;

pub use config::{Gdn2Config, Gdn2Mode};
pub use forward::{chunk_wy_forward, verify_chunk_vs_reference};
pub use l2norm::l2_normalize;
pub use module::{GatedDeltaNet2, ProjectedInputs, rms_norm_gate_per_head};
pub use short_conv::short_conv_1d;
