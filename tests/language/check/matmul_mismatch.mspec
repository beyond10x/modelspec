// expect E-TYPE-012 7:16 matmul contracting dimension mismatch
model Mismatch {
    param w: Tensor<f32, [64, 128]> = xavier_uniform();

    fn forward(x: Tensor<f32, [64, 32]>)
        -> Tensor<f32, [64, 128]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Mismatch;
    input x: Tensor<f32, [64, 32]>;
    target y: Tensor<f32, [64, 128]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 10;
    seed = 42;
    device = cpu;
}
