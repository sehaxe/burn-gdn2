#[test]
#[cfg(feature = "binary-tests")]
fn test_chunk_vs_reference() {
    use std::io::{Cursor, Read};
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use burn::module::Param;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::{Tensor, TensorData};
    use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2};

    fn read_i32(c: &mut Cursor<&[u8]>) -> i32 {
        let mut buf = [0u8; 4]; c.read_exact(&mut buf).unwrap(); i32::from_le_bytes(buf)
    }
    fn read_bool(c: &mut Cursor<&[u8]>) -> bool {
        let mut buf = [0u8; 1]; c.read_exact(&mut buf).unwrap(); buf[0] != 0
    }
    fn read_name(c: &mut Cursor<&[u8]>) -> String {
        let len = read_i32(c) as usize;
        let mut buf = vec![0u8; len]; c.read_exact(&mut buf).unwrap(); String::from_utf8(buf).unwrap()
    }
    fn read_f32_tensor(c: &mut Cursor<&[u8]>) -> (String, Vec<usize>, Vec<f32>) {
        let name = read_name(c);
        let ndim = read_i32(c) as usize;
        let size = read_i32(c) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim { shape.push(read_i32(c) as usize); }
        let mut flat = vec![0f32; size];
        let nbytes = size * 4;
        let buf = unsafe { std::slice::from_raw_parts_mut(flat.as_mut_ptr() as *mut u8, nbytes) };
        c.read_exact(buf).unwrap();
        (name, shape, flat)
    }
    fn read_raw_f32_tensor(c: &mut Cursor<&[u8]>) -> (Vec<usize>, Vec<f32>) {
        let ndim = read_i32(c) as usize;
        let size = read_i32(c) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim { shape.push(read_i32(c) as usize); }
        let mut flat = vec![0f32; size];
        let nbytes = size * 4;
        let buf = unsafe { std::slice::from_raw_parts_mut(flat.as_mut_ptr() as *mut u8, nbytes) };
        c.read_exact(buf).unwrap();
        (shape, flat)
    }
    fn t2(flat: &[f32], shape: &[usize], device: &NdArrayDevice) -> Tensor<NdArray, 2> {
        Tensor::from_data(TensorData::new(flat.to_vec(), shape.to_vec()), device)
    }
    fn t1(flat: &[f32], shape: &[usize], device: &NdArrayDevice) -> Tensor<NdArray, 1> {
        Tensor::from_data(TensorData::new(flat.to_vec(), shape.to_vec()), device)
    }
    fn lin_w(weight: Tensor<NdArray, 2>, device: &NdArrayDevice) -> Linear<NdArray> {
        let [out_f, in_f] = weight.shape().dims::<2>();
        let mut lin = LinearConfig::new(in_f, out_f).with_bias(false).init(device);
        lin.weight = Param::from_tensor(weight); lin
    }
    fn lin_wb(weight: Tensor<NdArray, 2>, bias: Tensor<NdArray, 1>, device: &NdArrayDevice) -> Linear<NdArray> {
        let [out_f, in_f] = weight.shape().dims::<2>();
        let mut lin = LinearConfig::new(in_f, out_f).with_bias(true).init(device);
        lin.weight = Param::from_tensor(weight);
        lin.bias = Some(Param::from_tensor(bias)); lin
    }

    let data = include_bytes!("ref_data.bin");
    let mut c = Cursor::new(data.as_slice());

    let d = read_i32(&mut c) as usize;
    let h = read_i32(&mut c) as usize;
    let hk = read_i32(&mut c) as usize;
    let hv = read_i32(&mut c) as usize;
    let expand_v = read_i32(&mut c) as f32 / 10.0;
    let use_short_conv = read_bool(&mut c);
    let allow_neg_eigval = read_bool(&mut c);

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for _ in 0..17 { tensors.push(read_f32_tensor(&mut c)); }
    let n_cases = read_i32(&mut c) as usize;

    let device = NdArrayDevice::Cpu;
    let get = |name: &str| -> &[f32] { let (_, _, d) = tensors.iter().find(|(n, _, _)| n == name).unwrap(); d };
    let get_shape = |name: &str| -> &[usize] { let (_, s, _) = tensors.iter().find(|(n, _, _)| n == name).unwrap(); s };
    let mk = |name| t2(get(name), get_shape(name), &device);
    let mk1 = |name| t1(get(name), get_shape(name), &device);

    let pos = c.position();

    let mut global_max_diff: f32 = 0.0;
    let mut n_fail = 0;
    let epsilon = 1e-3;

    for chunk_size in [4, 8, 16, 32, 64] {
        let cfg = Gdn2Config {
            hidden_size: d, num_heads: h, head_dim: hk,
            num_v_heads: Some(hv), expand_v, use_short_conv, allow_neg_eigval,
            norm_eps: 1e-5, mode: Gdn2Mode::Chunk, chunk_size,
        };
        let module = GatedDeltaNet2 {
            q_proj: lin_w(mk("q_proj"), &device),
            k_proj: lin_w(mk("k_proj"), &device),
            v_proj: lin_w(mk("v_proj"), &device),
            f_proj_0: lin_w(mk("f_proj_0"), &device),
            f_proj_1: lin_w(mk("f_proj_1"), &device),
            b_proj: lin_w(mk("b_proj"), &device),
            w_proj: lin_w(mk("w_proj"), &device),
            g_proj_0: lin_w(mk("g_proj_0"), &device),
            g_proj_1: lin_wb(mk("g_proj_1_w"), mk1("g_proj_1_b"), &device),
            a_log: Param::from_tensor(mk1("A_log")),
            dt_bias: Param::from_tensor(mk1("dt_bias")),
            o_norm_weight: Param::from_tensor(mk1("o_norm_w")),
            o_proj: lin_w(mk("o_proj"), &device),
            q_conv_w: Param::from_tensor(mk("q_conv_w")),
            k_conv_w: Param::from_tensor(mk("k_conv_w")),
            v_conv_w: Param::from_tensor(mk("v_conv_w")),
            config: cfg,
        };

        c.set_position(pos);

        let mut case_diff: f32 = 0.0;
        for i in 0..n_cases {
            let (in_shape, in_data) = read_raw_f32_tensor(&mut c);
            let (out_shape, out_data) = read_raw_f32_tensor(&mut c);
            let input = Tensor::<NdArray, 3>::from_data(TensorData::new(in_data, in_shape), &device);
            let ref_out = Tensor::<NdArray, 3>::from_data(TensorData::new(out_data, out_shape), &device);
            let mut state: Option<Tensor<NdArray, 4>> = None;
            let output = module.forward(input, &mut state, true);
            let out_bytes: Vec<f32> = output.into_data().bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let ref_bytes: Vec<f32> = ref_out.into_data().bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let diff = out_bytes.iter().zip(ref_bytes.iter())
                .map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
            case_diff = case_diff.max(diff);
            if diff >= epsilon {
                n_fail += 1;
                if n_fail <= 3 {
                    eprintln!("  FAIL chunk_size={chunk_size} case[{i}] diff={diff:.2e}");
                }
            }
        }
        global_max_diff = global_max_diff.max(case_diff);
        println!("  chunk_size={chunk_size:>2}: max_diff = {case_diff:.2e}{}",
            if case_diff < epsilon { " OK" } else { " FAIL" });
    }

    println!("Chunk all sizes: max_diff = {global_max_diff:.2e}, failures = {n_fail}");
    assert!(n_fail == 0, "{n_fail} failures");
    assert!(global_max_diff < 1e-2, "global max_diff too large: {global_max_diff:.2e}");
}
