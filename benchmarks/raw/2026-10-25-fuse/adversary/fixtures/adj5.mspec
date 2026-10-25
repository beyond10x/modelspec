model Adv {
    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
    param w2: Tensor<f32, [9, 5]> = xavier_uniform();
    param w3: Tensor<f32, [9, 5]> = xavier_uniform();
    param w4: Tensor<f32, [9, 5]> = xavier_uniform();
    param w5: Tensor<f32, [9, 5]> = xavier_uniform();
    param w6: Tensor<f32, [9, 5]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [10, 5]> {
        let h = relu(matmul(x, w1));
        return add(add(add(add(matmul(h, w2), matmul(h, w3)), matmul(h, w4)), matmul(h, w5)), matmul(h, w6));
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [10, 5]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 200;
    seed = 7;
    device = vulkan(0);
}
