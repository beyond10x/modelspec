model Dense {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param b: Tensor<f32, [3]> = zeros();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return gelu(add_rows(matmul(x, w), b));
    }
}

train Fit {
    model = Dense;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 20;
    seed = 7;
    device = cpu;
}
