// expect E-TYPE-023 15:55 `beta1` of `adamw` must be in [0, 1); got 1.0
model Reg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = adamw(lr = 0.05, weight_decay = 0.01, beta1 = 1.0, beta2 = 0.999, eps = 1e-8, warmup_steps = 2);
    steps = 10;
    seed = 7;
    device = cpu;
}
