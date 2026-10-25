model Norm {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param gamma: Tensor<f32, [1, 3]> = ones();
    param beta: Tensor<f32, [3]> = zeros();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return layer_norm(matmul(x, w), gamma, beta);
    }
}

train Fit {
    model = Norm;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
