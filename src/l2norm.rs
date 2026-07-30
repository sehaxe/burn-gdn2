use burn::tensor::{backend::Backend, Tensor};

/// L2-normalize along the last dimension.
pub fn l2_normalize<B: Backend>(x: Tensor<B, 3>, eps: f64) -> Tensor<B, 3> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(2).add_scalar(eps).sqrt();
    x.div(norm)
}
