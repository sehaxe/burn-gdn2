/// Operational mode for the GDN-2 recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn2Mode {
    /// Token-by-token scan. Best for inference (autoregressive, state-passing).
    /// The state `S [B, H, K, V]` is carried across tokens and updated in place.
    FusedRecurrent,
    /// Chunked scan with WY representation. Best for training.
    /// Splits the sequence into chunks of `chunk_size` and handles intra-chunk
    /// interactions via dense matmuls using a WY-like block decomposition.
    /// Complexity stays linear in total sequence length.
    Chunk,
}

/// Configuration for a [`GatedDeltaNet2`](crate::GatedDeltaNet2) layer.
///
/// GDN-2 decouples the erase and write operations of the gated delta rule
/// into two independent channel-wise gates:
///   - `b` (erase gate, `[0,1]^{d_k}`) - how much of each key channel to erase
///   - `w` (write gate, `[0,1]^{d_v}`) - how much of each value channel to write
///
/// Setting `b_t = β·1` and `w_t = β·1` recovers scalar-gated delta rule.
/// Further collapsing the per-channel decay to a scalar recovers Gated DeltaNet.
#[derive(Debug, Clone)]
pub struct Gdn2Config {
    /// Hidden size of the input (model dimension).
    pub hidden_size: usize,
    /// Number of query/key heads.
    pub num_heads: usize,
    /// Dimension per key/query head.
    pub head_dim: usize,
    /// Expansion factor for the value dimension.
    /// `head_v_dim = head_dim * expand_v`, `value_dim = num_v_heads * head_v_dim`.
    pub expand_v: f32,
    /// Number of value heads. If `None`, equals `num_heads`.
    /// GVA (Grouped Value Attention) is applied when `num_v_heads > num_heads`.
    pub num_v_heads: Option<usize>,
    /// Whether to apply a short-depthwise convolution before the recurrence.
    pub use_short_conv: bool,
    /// Allow negative eigenvalues in the state transition by lifting the
    /// erase gate range from `[0, 1]` to `[0, 2]`.
    /// See: "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"
    pub allow_neg_eigval: bool,
    /// Epsilon for output normalization.
    pub norm_eps: f64,
    /// Forward mode: fused-recurrent (inference) or chunk (training).
    pub mode: Gdn2Mode,
    /// Chunk size for `Chunk` mode. Default: 64 (matching reference).
    pub chunk_size: usize,
    /// Lower bound on per-channel decay.
    /// `None` (default) = standard GDN2 with unbounded decay (0..1).
    /// `Some(0.9)` = guaranteed minimum decay per channel. Adds learned
    /// per-channel factors: `min_decay + (1-min_decay)·sigmoid(w)`.
    /// Improves long-range memory at the cost of slightly less adaptivity.
    pub min_decay: Option<f64>,
}

impl Default for Gdn2Config {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_heads: 16,
            head_dim: 128,
            expand_v: 1.0,
            num_v_heads: None,
            use_short_conv: true,
            allow_neg_eigval: false,
            norm_eps: 1e-5,
            mode: Gdn2Mode::FusedRecurrent,
            chunk_size: 64,
            min_decay: None,
        }
    }
}
