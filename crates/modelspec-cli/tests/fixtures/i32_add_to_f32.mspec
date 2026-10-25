model AddIds {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [4, 1]>, x: Tensor<f32, [4, 1]>) -> Tensor<f32, [4, 1]> {
        return add(ids, x);
    }
}

train Fit {
    model = AddIds;
    input ids: Tensor<i32, [4, 1]>;
    input x: Tensor<f32, [4, 1]>;
    target y: Tensor<f32, [4, 1]>;
    loss = mse(model.forward(ids, x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
