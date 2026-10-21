// expect E-SYNTAX-008 3:27 symbolic dimension `B`
model M {
    param w: Tensor<f32, [B, 1]> = zeros();
    fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> {
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
