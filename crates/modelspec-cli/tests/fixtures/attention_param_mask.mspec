model Attend {
    param w_q: Tensor<f32, [4, 4]> = xavier_uniform();
    param w_k: Tensor<f32, [4, 4]> = xavier_uniform();
    param w_v: Tensor<f32, [4, 4]> = xavier_uniform();
    param key_mask: Tensor<f32, [2, 3]> = ones();
    fn forward(x: Tensor<f32, [6, 4]>) -> Tensor<f32, [6, 4]> {
        return attention(matmul(x, w_q), matmul(x, w_k), matmul(x, w_v), key_mask, heads = 2);
    }
}

train Fit {
    model = Attend;
    input x: Tensor<f32, [6, 4]>;
    target y: Tensor<f32, [6, 4]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 7;
    device = cpu;
}
