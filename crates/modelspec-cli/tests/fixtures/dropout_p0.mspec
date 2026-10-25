model Reg {
    param w1: Tensor<f32, [4, 6]> = xavier_uniform();
    param w2: Tensor<f32, [6, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let h = relu(matmul(x, w1));
        let d = dropout(h, p = 0.0);
        return matmul(d, w2);
    }
}

train Fit {
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 11;
    device = cpu;
}
