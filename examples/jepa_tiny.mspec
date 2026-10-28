// A LeJEPA-style joint-embedding predictive model at toy size. `context` and `view` are two views
// of the same 32 samples, for example a context crop and a target crop. One encoder (`enc_w`,
// `enc_b`) embeds both; a predictor (`pred_w`, `pred_b`) maps the context embedding onto the
// view's. The prediction is compared with the view's embedding under `stop_gradient`, so the
// target branch passes no gradient, and `sigreg` holds the context embeddings to an isotropic
// Gaussian, which keeps them from collapsing. Synthetic data feeds `context` and `view` without
// `--data`; a dense directory feeds them as `context.f32` and `view.f32`.
model Jepa {
    param enc_w: Tensor<f32, [16, 16]> = xavier_uniform();
    param enc_b: Tensor<f32, [16]> = zeros();
    param pred_w: Tensor<f32, [16, 16]> = xavier_uniform();
    param pred_b: Tensor<f32, [16]> = zeros();

    fn encode(view: Tensor<f32, [32, 16]>) -> Tensor<f32, [32, 16]> {
        return gelu(add_rows(matmul(view, enc_w), enc_b));
    }

    fn forward(context: Tensor<f32, [32, 16]>) -> Tensor<f32, [32, 16]> {
        let z = gelu(add_rows(matmul(context, enc_w), enc_b));
        return add_rows(matmul(z, pred_w), pred_b);
    }
}

train Fit {
    model = Jepa;
    input context: Tensor<f32, [32, 16]>;
    target view: Tensor<f32, [32, 16]>;
    loss = add(mse(model.forward(context), stop_gradient(model.encode(view))), mul(sigreg(model.encode(context), slices = 8), 0.1));
    optimizer = sgd(lr = 0.05);
    steps = 200;
    seed = 7;
    device = cpu;
}
