// expect E-TYPE-013 3:36 initializer `normal(...)` is not supported; the initializers are `xavier_uniform()`, `zeros()` and `ones()`
model LinearReg {
    param w: Tensor<f32, [4, 1]> = normal();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
