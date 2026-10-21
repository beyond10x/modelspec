// expect E-TYPE-017 14:30 the loss of train block `Fit` must call `model.forward` exactly once; it calls it 2 times
model Square {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 4]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Square;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 4]>;
    loss = mse(model.forward(model.forward(x)), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
