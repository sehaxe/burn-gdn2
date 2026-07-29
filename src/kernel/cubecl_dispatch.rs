//! High-level dispatch for CubeCL-accelerated GDN2 kernels.
//!
//! These functions operate on raw [`Handle`]s and are generic over any
//! [`Runtime`].  The caller is responsible for extracting handles from
//! their tensor representation (e.g. `burn_cubecl::CubeTensor`) and
//! wrapping the output handles back.
//!
//! # Safety
//!
//! All handles must point to contiguous row-major buffers of the correct
//! size.  Shape parameters must match the actual buffer contents.
//! Overlapping input/output handles produce undefined behaviour.

use cubecl::client::ComputeClient;
use cubecl::server::Handle;
use cubecl::Runtime;

/// Launch the fused recurrent GDN-2 scan on a CubeCL device.
///
/// **Launch config:**
/// - Grid size `(n_heads_q, n_batch, 1)`
/// - Cube (block) size `(dim_v, 1, 1)` — one thread per V-dim element
///
/// **Buffer sizes (elements, not bytes):**
/// - `q, k, g, b`: `n_batch × n_heads_q × n_tokens × dim_k`
/// - `v, w, out`:  `n_batch × n_heads_v × n_tokens × dim_v`
/// - `state`:      `n_batch × n_heads_v × dim_k × dim_v`
///
/// # Safety
///
/// See module-level safety docs.
#[cfg(feature = "cubecl")]
pub unsafe fn run_fused_recurrent<R: Runtime>(
    client: &ComputeClient<R>,
    n_batch: u32,
    n_heads_q: u32,
    n_tokens: u32,
    dim_k: u32,
    dim_v: u32,
    hv: u32,
    scale: f32,
    q: &Handle,
    k: &Handle,
    v: &Handle,
    g: &Handle,
    b: &Handle,
    w: &Handle,
    out: &Handle,
    state: &Handle,
) {
    // Safety: caller guarantees handle sizes and shapes.
    super::cubecl_fused::launch_fused_recurrent::<R>(
        client, q, k, v, g, b, w, out, state, scale, n_batch, n_heads_q,
        n_tokens, dim_k, dim_v, hv,
    );
}

/// Launch the chunk forward‑substitution solve on a CubeCL device.
///
/// Solves `(I − Akk)⁻¹ × [rhs_k | rhs_v] → [w_wy | u]` for every chunk
/// independently.
///
/// **Launch config:**
/// - Grid `(n_chunks, n_batch × n_heads_v, 1)`
/// - Cube (block) `(1, 1, 1)` — single‑threaded solve per chunk
///
/// **Buffer sizes (elements):**
/// - `akk`:  `n_batch × n_heads_v × n_chunks × chunk_size × chunk_size`
/// - `rhs_k, w_wy`: `n_batch × n_heads_v × n_chunks × chunk_size × dim_k`
/// - `rhs_v, u`:    `n_batch × n_heads_v × n_chunks × chunk_size × dim_v`
///
/// # Safety
///
/// See module-level safety docs.
#[cfg(feature = "cubecl")]
pub unsafe fn run_chunk_substitution<R: Runtime>(
    client: &ComputeClient<R>,
    n_batch: u32,
    n_heads_v: u32,
    n_chunks: u32,
    chunk_size: u32,
    dim_k: u32,
    dim_v: u32,
    akk: &Handle,
    rhs_k: &Handle,
    rhs_v: &Handle,
    w_wy: &Handle,
    u: &Handle,
) {
    // Safety: caller guarantees handle sizes and shapes.
    super::cubecl_chunk::launch_chunk_substitution::<R>(
        client, akk, rhs_k, rhs_v, w_wy, u, n_batch, n_heads_v, n_chunks,
        chunk_size, dim_k, dim_v,
    );
}
