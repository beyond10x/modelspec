model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = adamw(warmup_steps = 0, eps = 1e-6, beta2 = 0.98, beta1 = 0.8, weight_decay = 0.0, lr = 0.01);
    steps = 200;
    seed = 42;
    device = cpu;
}
