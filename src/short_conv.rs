use burn::tensor::activation::silu;
use burn::tensor::{backend::Backend, Tensor};

/// 1D causal depthwise convolution with kernel_size=4.
pub fn short_conv_1d<B: Backend>(x: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    let [b, t, c] = x.shape().dims();
    let first = x.clone().slice([0..b, 0..1, 0..c]);
    let pad = first.repeat(&[1, 3, 1]);
    let device = x.device();
    let x_pad = Tensor::cat(vec![pad, x], 1);
    let mut out = Tensor::zeros([b, t, c], &device);
    for i in 0..4 {
        let x_slice = x_pad.clone().slice([0..b, i..i + t, 0..c]);
        let w_i = weight.clone().slice([0..c, i..i + 1]).reshape([1, 1, c]);
        out = out + x_slice * w_i;
    }
    silu(out)
}
