model MlmBag {
    param table: Tensor<f32, [300, 16]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [64]>) -> Tensor<f32, [64, 300]> {
        return matmul(dropout(embedding(table, ids), p = 0.1), transpose(table));
    }
}

train Fit {
    model = MlmBag;
    input ids: Tensor<i32, [64]>;
    target labels: Tensor<i32, [64]>;
    target weights: Tensor<f32, [64]>;
    loss = cross_entropy(model.forward(ids), labels, weights);
    optimizer = sgd(lr = 1.0);
    steps = 200;
    seed = 7;
    device = cpu;
}
