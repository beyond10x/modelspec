model Dense {
    param w: Tensor<f32, [4, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 4]>)
        -> Tensor<f32, [16, 8]> {
        return matmul(x, w);
    }
}

train FitDense {
    model = Dense;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, [16, 8]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 100;
    seed = 7;
    device = cpu;
}
