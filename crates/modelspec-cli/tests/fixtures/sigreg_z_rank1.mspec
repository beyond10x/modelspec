model Spread {
    param w: Tensor<f32, [1, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16]>) -> Tensor<f32, []> {
        let s = sigreg(x, slices = 4);
        return add(s, sum(w));
    }
}

train Fit {
    model = Spread;
    input x: Tensor<f32, [16]>;
    target y: Tensor<f32, []>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
