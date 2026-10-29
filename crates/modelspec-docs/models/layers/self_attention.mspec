model SelfAttention {
    param wq: Tensor<f32, [8, 8]> = xavier_uniform();
    param wk: Tensor<f32, [8, 8]> = xavier_uniform();
    param wv: Tensor<f32, [8, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 8]>, key_mask: Tensor<f32, [2, 8]>)
        -> Tensor<f32, [16, 8]> {
        let q = matmul(x, wq);
        let k = matmul(x, wk);
        let v = matmul(x, wv);
        return attention(q, k, v, key_mask, heads = 2);
    }
}

train FitSelfAttention {
    model = SelfAttention;
    input x: Tensor<f32, [16, 8]>;
    input key_mask: Tensor<f32, [2, 8]>;
    target y: Tensor<f32, [16, 8]>;
    loss = mse(model.forward(x, key_mask), y);
    optimizer = sgd(lr = 0.05);
    steps = 100;
    seed = 7;
    device = cpu;
}
