// expect E-TYPE-022 6:38 `p` of `dropout` must be in [0, 1); got 1.0
model Drop {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [6, 4]>) -> Tensor<f32, [6, 4]> {
        return dropout(matmul(x, w), p = 1.0);
    }
}

train Fit {
    model = Drop;
    input x: Tensor<f32, [6, 4]>;
    target y: Tensor<f32, [6, 4]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
