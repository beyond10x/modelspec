// expect E-TYPE-014 4:33 `xavier_uniform()` needs a rank-2 parameter; `b` has type Tensor<f32, [4]>
model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    param b: Tensor<f32, [4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
