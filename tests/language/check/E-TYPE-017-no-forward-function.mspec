// expect E-TYPE-017 14:12 the loss of train block `Fit` must call `model.forward` exactly once; model `LinearReg` has no function `forward`
model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn predict(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.predict(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
