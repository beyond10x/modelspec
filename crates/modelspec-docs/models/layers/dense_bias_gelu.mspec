model DenseBiasGelu {
    param w: Tensor<f32, [4, 8]> = xavier_uniform();
    param b: Tensor<f32, [8]> = zeros();

    fn forward(x: Tensor<f32, [16, 4]>)
        -> Tensor<f32, [16, 8]> {
        return gelu(add_rows(matmul(x, w), b));
    }
}

train FitDenseBiasGelu {
    model = DenseBiasGelu;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, [16, 8]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 100;
    seed = 7;
    device = cpu;
}
