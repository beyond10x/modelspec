model Dense {
    param w: Tensor<f32, [3, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [3, 3]>, b: Tensor<f32, [3]>) -> Tensor<f32, [3, 3]> {
        return add_rows(matmul(x, w), b);
    }
}

train Fit {
    model = Dense;
    input x: Tensor<f32, [3, 3]>;
    input b: Tensor<f32, [3]>;
    target y: Tensor<f32, [3, 3]>;
    loss = mse(model.forward(x, b), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
