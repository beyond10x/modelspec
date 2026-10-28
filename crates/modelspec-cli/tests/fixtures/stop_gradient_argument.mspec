model Target {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();
    param v: Tensor<f32, [4, 4]> = xavier_uniform();

    fn encode(x: Tensor<f32, [6, 4]>) -> Tensor<f32, [6, 4]> {
        return matmul(x, w);
    }

    fn forward(h: Tensor<f32, [6, 4]>) -> Tensor<f32, [6, 4]> {
        return matmul(stop_gradient(h), v);
    }
}

train Fit {
    model = Target;
    input x: Tensor<f32, [6, 4]>;
    target y: Tensor<f32, [6, 4]>;
    loss = mse(model.forward(model.encode(x)), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
