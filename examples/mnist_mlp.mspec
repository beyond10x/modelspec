// The shapes of the MNIST benchmark: 784 -> 64 relu -> 10, no bias, f32, MSE, SGD lr 0.1,
// batch 100, 1,800 steps. The data is M0 synthetic teacher data until a data interface exists.
model MnistMlp {
    param w1: Tensor<f32, [784, 64]> = xavier_uniform();
    param w2: Tensor<f32, [64, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [100, 784]>)
        -> Tensor<f32, [100, 10]> {
        return matmul(relu(matmul(x, w1)), w2);
    }
}

train Fit {
    model = MnistMlp;
    input x: Tensor<f32, [100, 784]>;
    target y: Tensor<f32, [100, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 1800;
    seed = 42;
    device = vulkan(0);
}
