model Adv {
    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
    param w2: Tensor<f32, [9, 5]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [10, 5]> {
        let d = sub(matmul(x, w1), 1.0);
        let e = mul(d, 2.0);
        return matmul(add(e, d), w2);
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [10, 5]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.002);
    steps = 200;
    seed = 7;
    device = vulkan(0);
}
