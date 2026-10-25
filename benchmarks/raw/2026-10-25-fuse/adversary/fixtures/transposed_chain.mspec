model Adv {
    param w1: Tensor<f32, [10, 7]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [12, 7]> {
        return mul(matmul(transpose(x), w1), 0.5);
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 7]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 200;
    seed = 7;
    device = vulkan(0);
}
