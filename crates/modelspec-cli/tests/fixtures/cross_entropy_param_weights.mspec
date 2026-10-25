model Mlm {
    param table: Tensor<f32, [5, 4]> = xavier_uniform();
    param scale: Tensor<f32, [6]> = ones();

    fn forward(ids: Tensor<i32, [6]>) -> Tensor<f32, [6, 5]> {
        return matmul(embedding(table, ids), transpose(table));
    }

    fn scaled(a: Tensor<f32, [6]>) -> Tensor<f32, [6]> {
        return mul(a, scale);
    }
}

train Fit {
    model = Mlm;
    input ids: Tensor<i32, [6]>;
    target labels: Tensor<i32, [6]>;
    target weights: Tensor<f32, [6]>;
    loss = cross_entropy(model.forward(ids), labels, model.scaled(weights));
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
