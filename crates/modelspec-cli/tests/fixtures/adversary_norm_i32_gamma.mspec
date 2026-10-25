model Norm {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param beta: Tensor<f32, [3]> = zeros();

    fn forward(ids: Tensor<i32, [3]>, x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return layer_norm(matmul(x, w), ids, beta);
    }
}

train Fit {
    model = Norm;
    input ids: Tensor<i32, [3]>;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(ids, x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
