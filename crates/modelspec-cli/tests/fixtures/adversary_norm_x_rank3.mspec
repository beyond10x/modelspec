model Norm {
    param gamma: Tensor<f32, [4]> = ones();
    param beta: Tensor<f32, [4]> = zeros();

    fn forward(x: Tensor<f32, [2, 3, 4]>) -> Tensor<f32, [2, 3, 4]> {
        return layer_norm(x, gamma, beta);
    }
}

train Fit {
    model = Norm;
    input x: Tensor<f32, [2, 3, 4]>;
    target y: Tensor<f32, [2, 3, 4]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
