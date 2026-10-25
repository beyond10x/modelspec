// expect E-TYPE-025 15:33 `momentum` of `muon` must be in [0, 1); got 1.0
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
    optimizer = muon(lr = 0.05, momentum = 1.0, weight_decay = 0.01, ns_steps = 5, warmup_steps = 2, adam_lr = 0.05, beta1 = 0.9, beta2 = 0.999, eps = 1e-8);
    steps = 10;
    seed = 7;
    device = cpu;
}
