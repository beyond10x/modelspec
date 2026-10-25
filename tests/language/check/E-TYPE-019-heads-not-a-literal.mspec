// expect E-TYPE-019 6:49 `heads` of `attention` takes an integer literal of at least 1
model Attend {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [6, 4]>, m: Tensor<f32, [2, 3]>) -> Tensor<f32, [6, 4]> {
        return attention(matmul(x, w), x, x, m, heads = 0);
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
