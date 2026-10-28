model Spread {
    param z: Tensor<f32, [256, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [256, 8]>) -> Tensor<f32, [256, 8]> {
        return add(z, mul(x, 0.0));
    }
}

train Fit {
    model = Spread;
    input x: Tensor<f32, [256, 8]>;
    target y: Tensor<f32, [256, 8]>;
    loss = add(sigreg(model.forward(x), slices = 16), mul(mse(x, y), 0.0));
    optimizer = sgd(lr = 0.5);
    steps = 100;
    seed = 5;
    device = cpu;
}
