model Joint {
    param w: Tensor<f32, [8, 8]> = xavier_uniform();
    param v: Tensor<f32, [8, 4]> = xavier_uniform();

    fn encode(x: Tensor<f32, [32, 8]>) -> Tensor<f32, [32, 8]> {
        return matmul(x, w);
    }

    fn forward(x: Tensor<f32, [32, 8]>) -> Tensor<f32, [32, 4]> {
        return matmul(x, v);
    }
}

train Fit {
    model = Joint;
    input x: Tensor<f32, [32, 8]>;
    target y: Tensor<f32, [32, 4]>;
    loss = add(add(sigreg(stop_gradient(model.encode(x)), slices = 4), mse(model.forward(x), y)), mul(mean(model.encode(x)), 0.0));
    optimizer = sgd(lr = 0.1);
    steps = 20;
    seed = 11;
    device = cpu;
}
