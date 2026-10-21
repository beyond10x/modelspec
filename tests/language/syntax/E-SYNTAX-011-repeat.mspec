// expect E-SYNTAX-011 5:9 `repeat`
model M {
    param w: Tensor<f32, [2, 1]> = zeros();
    fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> {
        repeat 2 { }
        return matmul(x, w);
    }
}
train T {
    model = M;
    input x: Tensor<f32, [3, 2]>;
    target y: Tensor<f32, [3, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 10;
    seed = 1;
    device = cpu;
}
