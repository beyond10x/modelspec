model PassIds {
    param table: Tensor<f32, [6, 3]> = xavier_uniform();

    fn ids_of(ids: Tensor<i32, [4]>) -> Tensor<i32, [4]> {
        return ids;
    }

    fn forward(ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 3]> {
        return embedding(table, ids);
    }
}

train Fit {
    model = PassIds;
    input ids: Tensor<i32, [4]>;
    target y: Tensor<f32, [4, 3]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
