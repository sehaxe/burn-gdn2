use burn::tensor::{backend::Backend, Tensor};

/// L2-normalize the last dimension of a 3D tensor.
///
/// `x / sqrt(sum(x^2, dim=-1) + eps)`
pub fn l2_normalize<B: Backend>(x: Tensor<B, 3>, eps: f64) -> Tensor<B, 3> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(2).add_scalar(eps).sqrt();
    x.div(norm)
}
