model LayerNormResidual {
    param w: Tensor<f32, [8, 8]> = xavier_uniform();
    param gamma: Tensor<f32, [8]> = ones();
    param beta: Tensor<f32, [8]> = zeros();

    fn forward(x: Tensor<f32, [16, 8]>)
        -> Tensor<f32, [16, 8]> {
        let f = matmul(x, w);
        return layer_norm(add(x, f), gamma, beta);
    }
}

train FitLayerNormResidual {
    model = LayerNormResidual;
    input x: Tensor<f32, [16, 8]>;
    target y: Tensor<f32, [16, 8]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 100;
    seed = 7;
    device = cpu;
}
