model Mlm {
    param table: Tensor<f32, [5, 4]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [6]>) -> Tensor<f32, [6, 5]> {
        return matmul(embedding(table, ids), transpose(table));
    }
}

train Fit {
    model = Mlm;
    input ids: Tensor<i32, [6]>;
    target labels: Tensor<f32, [6]>;
    target weights: Tensor<f32, [6]>;
    loss = cross_entropy(model.forward(ids), labels, weights);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
