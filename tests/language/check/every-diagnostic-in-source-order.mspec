// expect E-TYPE-012 8:17 matmul contracting dimension mismatch
// expect E-NAME-001 9:26 unknown name `u`
// expect E-TYPE-011 19:5 `steps` must be positive; got 0
model LinearReg {
    param w: Tensor<f32, [3, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let a = matmul(x, w);
        return matmul(x, u);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 0;
    seed = 42;
    device = cpu;
}
