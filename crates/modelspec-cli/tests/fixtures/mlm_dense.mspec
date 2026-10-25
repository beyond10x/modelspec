model MlmDense {
    param table: Tensor<f32, [300, 16]> = xavier_uniform();
    param w1: Tensor<f32, [16, 16]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [64]>) -> Tensor<f32, [64, 300]> {
        return matmul(gelu(matmul(embedding(table, ids), w1)), transpose(table));
    }
}

train Fit {
    model = MlmDense;
    input ids: Tensor<i32, [64]>;
    target labels: Tensor<i32, [64]>;
    target weights: Tensor<f32, [64]>;
    loss = cross_entropy(model.forward(ids), labels, weights);
    optimizer = muon(lr = 0.01, momentum = 0.95, weight_decay = 0.01, ns_steps = 5, warmup_steps = 30, adam_lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8);
    steps = 300;
    seed = 7;
    device = cpu;
}
