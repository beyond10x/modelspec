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
    optimizer = muon(lr = 0.05, momentum = 0.95, weight_decay = 0.01, ns_steps = 5, warmup_steps = 10, adam_lr = 0.05, beta1 = 0.9, beta2 = 0.999, eps = 1e-8);
    steps = 200;
    seed = 42;
    device = cpu;
}
