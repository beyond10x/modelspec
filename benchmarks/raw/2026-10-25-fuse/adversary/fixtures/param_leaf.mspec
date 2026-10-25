model Adv {
    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
    param s: Tensor<f32, [10, 9]> = ones();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [10, 9]> {
        return add(mul(sub(matmul(x, w1), s), 0.5), s);
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [10, 9]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 200;
    seed = 7;
    device = vulkan(0);
}
