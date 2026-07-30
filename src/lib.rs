//! # burn-gdn2
//!
//! Gated DeltaNet 2 (GDN-2) — a linear‑complexity recurrent token mixer
//! with channel‑wise erase/write gates.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2};
//!
//! type Backend = burn_ndarray::NdArray;
//! let device = Default::default();
//! let config = Gdn2Config {
//!     hidden_size: 64,
//!     num_heads: 2,
//!     head_dim: 32,
//!     ..Default::default()
//! };
//! let model = GatedDeltaNet2::<Backend>::new(&config, &device);
//! ```
//!
//! ## Features
//!
//! - **`std`** (default) — standard library support
//! - **`autodiff`** — differentiation support (required for training)

pub mod config;
pub mod forward;
pub mod kernel;
pub mod l2norm;
pub mod module;
pub mod short_conv;

pub use config::{Gdn2Config, Gdn2Mode};
pub use forward::{chunk_wy_forward, verify_chunk_vs_reference};
pub use kernel::fused_recurrent::fused_recurrent_forward;
pub use l2norm::l2_normalize;
pub use module::{rms_norm_gate_per_head, GatedDeltaNet2, ProjectedInputs};
pub use short_conv::short_conv_1d;
