model Spread {
    param w: Tensor<f32, [1, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 4]>) -> Tensor<f32, []> {
        let s = sigreg(x, slices = 18446744073709551615);
        return add(s, sum(w));
    }
}

train Fit {
    model = Spread;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, []>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
