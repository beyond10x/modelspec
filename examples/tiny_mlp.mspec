// ModelSpec target syntax (DESIGN.md v0.1); not yet implemented.
module Linear<In: dim, Out: dim> {
    param weight: Tensor<f32, [In, Out]> = xavier_uniform();

    fn forward<B: dim>(x: Tensor<f32, [B, In]>)
        -> Tensor<f32, [B, Out]> {
        return matmul(x, weight);
    }
}

model TinyMLP<B: dim> {
    layer fc1: Linear<32, 64>;
    layer fc2: Linear<64, 1>;

    fn forward(x: Tensor<f32, [B, 32]>)
        -> Tensor<f32, [B, 1]> {
        let h = relu(fc1.forward(x));
        return fc2.forward(h);
    }
}

train TinyFit {
    model = TinyMLP<64>;
    input  x: Tensor<f32, [64, 32]>;
    target y: Tensor<f32, [64, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 1000;
    seed = 42;
    device = vulkan(0);
}
