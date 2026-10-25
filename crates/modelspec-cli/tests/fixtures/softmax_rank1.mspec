model Probs {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param v: Tensor<f32, [3]> = zeros();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        let p = softmax(v);
        return add_rows(matmul(x, w), p);
    }
}

train Fit {
    model = Probs;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
