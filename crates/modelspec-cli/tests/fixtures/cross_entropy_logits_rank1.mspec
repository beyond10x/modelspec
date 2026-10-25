model Scores {
    param w: Tensor<f32, [6]> = ones();

    fn forward(x: Tensor<f32, [6]>) -> Tensor<f32, [6]> {
        return mul(x, w);
    }
}

train Fit {
    model = Scores;
    input x: Tensor<f32, [6]>;
    target labels: Tensor<i32, [6]>;
    target weights: Tensor<f32, [6]>;
    loss = cross_entropy(model.forward(x), labels, weights);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
