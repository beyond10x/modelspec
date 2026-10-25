// expect E-TYPE-021 7:46 `key_mask` of `attention` reads a model parameter
model Attend {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();
    param p: Tensor<f32, [2, 3]> = ones();

    fn forward(x: Tensor<f32, [6, 4]>, m: Tensor<f32, [2, 3]>) -> Tensor<f32, [6, 4]> {
        return attention(matmul(x, w), x, x, mul(m, p), heads = 2);
    }
}

train Fit {
    model = Attend;
    input x: Tensor<f32, [6, 4]>;
    input m: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [6, 4]>;
    loss = mse(model.forward(x, m), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
