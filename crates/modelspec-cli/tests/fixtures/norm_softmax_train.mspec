model Norm {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param gamma: Tensor<f32, [3]> = ones();
    param beta: Tensor<f32, [3]> = zeros();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return softmax(layer_norm(matmul(x, w), gamma, beta));
    }
}

train Fit {
    model = Norm;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 1.0);
    steps = 20;
    seed = 7;
    device = cpu;
}
