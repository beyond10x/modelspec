// expect E-TYPE-016 5:17 element or byte count of Tensor<f32, [4294967296, 4294967296]> overflows u64
model Big {
    param w: Tensor<f32, [1, 4294967296]> = xavier_uniform();
    fn forward(x: Tensor<f32, [4294967296, 1]>) -> Tensor<f32, []> {
        let p = matmul(x, w);
        return sum(p);
    }
}

train Fit {
    model = Big;
    input x: Tensor<f32, [4294967296, 1]>;
    target y: Tensor<f32, []>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
