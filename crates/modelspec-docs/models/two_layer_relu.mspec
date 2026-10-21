model TwoLayer {
    param w1: Tensor<f32, [4, 8]> = xavier_uniform();
    param w2: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 4]>)
        -> Tensor<f32, [16, 1]> {
        let h = relu(matmul(x, w1));
        return matmul(h, w2);
    }
}

train FitTwoLayer {
    model = TwoLayer;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, [16, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 200;
    seed = 7;
    device = cpu;
}
