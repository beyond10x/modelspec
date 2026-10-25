// expect E-TYPE-018 6:16 `mul` operand 1 must be f32
model Scale {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [8, 1]>) -> Tensor<f32, [8, 1]> {
        return mul(ids, 2.0);
    }
}

train Fit {
    model = Scale;
    input ids: Tensor<i32, [8, 1]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
