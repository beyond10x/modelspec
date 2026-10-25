model Probs {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [2, 3, 4]>) -> Tensor<f32, [2, 3, 4]> {
        return softmax(x);
    }
}

train Fit {
    model = Probs;
    input x: Tensor<f32, [2, 3, 4]>;
    target y: Tensor<f32, [2, 3, 4]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
