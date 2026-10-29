// The zoom fixture (story:diagram-zoom): the loss calls `forward` once and `head` twice, both
// `head` calls under `stop_gradient`, so the model view holds the scopes `forward`, `head` and
// `head@1`. `head` reads its argument twice, so the edge from `head` into `head@1` and the edge
// from `z` into `head` each stand for two operands. `forward` reads `v` too, so it has a gradient.
model Two {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param v: Tensor<f32, [3, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return relu(matmul(matmul(x, w), v));
    }

    fn head(h: Tensor<f32, [8, 3]>) -> Tensor<f32, [8, 3]> {
        return dropout(relu(add(matmul(h, v), h)), p = 0.1);
    }
}

train Fit {
    model = Two;
    input x: Tensor<f32, [8, 4]>;
    input z: Tensor<f32, [8, 3]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(add(model.forward(x), stop_gradient(model.head(model.head(z)))), y);
    optimizer = sgd(lr = 0.05);
    steps = 2;
    seed = 1;
    device = cpu;
}
