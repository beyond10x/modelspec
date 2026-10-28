// infer_scores.mspec: two functions over one weight (story:decider-infer-on-data). `forward`
// returns the loss, as the decider's does, so the train block's loss is the call itself and its
// target is a scalar; `scores` is the [B, C] linear map, B = 4 and C = 2, that
// `infer --function scores --data` writes one row per sample of.
model Scorer {
    param w: Tensor<f32, [3, 2]> = xavier_uniform();

    fn forward(x: Tensor<f32, [4, 3]>, y: Tensor<f32, []>) -> Tensor<f32, []> {
        let e = sub(matmul(x, w), y);
        return mean(mul(e, e));
    }

    fn scores(x: Tensor<f32, [4, 3]>) -> Tensor<f32, [4, 2]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Scorer;
    input x: Tensor<f32, [4, 3]>;
    target y: Tensor<f32, []>;
    loss = model.forward(x, y);
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 7;
    device = cpu;
}
