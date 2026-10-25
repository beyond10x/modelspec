// The MLP size sweep at batch 8192, hidden 4096 (benches/mlp_sweep.rs).
model SweepMlp {
    param w1: Tensor<f32, [784, 4096]> = xavier_uniform();
    param w2: Tensor<f32, [4096, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8192, 784]>)
        -> Tensor<f32, [8192, 10]> {
        return matmul(relu(matmul(x, w1)), w2);
    }
}

train Fit {
    model = SweepMlp;
    input x: Tensor<f32, [8192, 784]>;
    target y: Tensor<f32, [8192, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 1000;
    seed = 42;
    device = vulkan(0);
}
