// expect E-NAME-004 5:16 `w` is already bound
model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(w: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 4]> {
        return w;
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 4]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
