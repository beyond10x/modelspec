// The transformer-block benchmark: the two encoder layers of examples/bert_tiny.mspec, written
// out the same way (post-layer-norm, hidden size 128, 2 attention heads, a feed-forward width of
// 512 with the tanh-form GELU, sequences of 128) on batches of 8 sequences, without the rest of
// BERT-tiny. There is no token or position table, no embedding layer norm, no masked-language-
// model head and no dropout: the input x is the [B·128, 128] activation the first layer reads, the
// loss is the mean squared error of the second layer's output against a target of the same shape,
// and the optimizer is SGD at a constant rate, not AdamW. The data is the runner's synthetic
// batch: x standard normal, the key mask all ones (no key is masked), and the target the
// prediction of a teacher with this file's shapes. It differs from the two layers of BERT-tiny
// only where those take dropout and in the first layer reading x instead of the normalised
// embeddings. crates/modelspec-runtime/benches/transformer_block.rs runs it at batches of 8, 32
// and 128 sequences, and opponents/transformer_block/ is the same program in PyTorch.
model TransformerBlock {
    param l0_wq: Tensor<f32, [128, 128]> = xavier_uniform();
    param l0_bq: Tensor<f32, [128]> = zeros();
    param l0_wk: Tensor<f32, [128, 128]> = xavier_uniform();
    param l0_bk: Tensor<f32, [128]> = zeros();
    param l0_wv: Tensor<f32, [128, 128]> = xavier_uniform();
    param l0_bv: Tensor<f32, [128]> = zeros();
    param l0_wo: Tensor<f32, [128, 128]> = xavier_uniform();
    param l0_bo: Tensor<f32, [128]> = zeros();
    param l0_ln1_gamma: Tensor<f32, [128]> = ones();
    param l0_ln1_beta: Tensor<f32, [128]> = zeros();
    param l0_w1: Tensor<f32, [128, 512]> = xavier_uniform();
    param l0_b1: Tensor<f32, [512]> = zeros();
    param l0_w2: Tensor<f32, [512, 128]> = xavier_uniform();
    param l0_b2: Tensor<f32, [128]> = zeros();
    param l0_ln2_gamma: Tensor<f32, [128]> = ones();
    param l0_ln2_beta: Tensor<f32, [128]> = zeros();

    param l1_wq: Tensor<f32, [128, 128]> = xavier_uniform();
    param l1_bq: Tensor<f32, [128]> = zeros();
    param l1_wk: Tensor<f32, [128, 128]> = xavier_uniform();
    param l1_bk: Tensor<f32, [128]> = zeros();
    param l1_wv: Tensor<f32, [128, 128]> = xavier_uniform();
    param l1_bv: Tensor<f32, [128]> = zeros();
    param l1_wo: Tensor<f32, [128, 128]> = xavier_uniform();
    param l1_bo: Tensor<f32, [128]> = zeros();
    param l1_ln1_gamma: Tensor<f32, [128]> = ones();
    param l1_ln1_beta: Tensor<f32, [128]> = zeros();
    param l1_w1: Tensor<f32, [128, 512]> = xavier_uniform();
    param l1_b1: Tensor<f32, [512]> = zeros();
    param l1_w2: Tensor<f32, [512, 128]> = xavier_uniform();
    param l1_b2: Tensor<f32, [128]> = zeros();
    param l1_ln2_gamma: Tensor<f32, [128]> = ones();
    param l1_ln2_beta: Tensor<f32, [128]> = zeros();

    fn forward(x: Tensor<f32, [1024, 128]>, key_mask: Tensor<f32, [8, 128]>)
        -> Tensor<f32, [1024, 128]> {
        let q0 = add_rows(matmul(x, l0_wq), l0_bq);
        let k0 = add_rows(matmul(x, l0_wk), l0_bk);
        let v0 = add_rows(matmul(x, l0_wv), l0_bv);
        let a0 = add_rows(matmul(attention(q0, k0, v0, key_mask, heads = 2), l0_wo), l0_bo);
        let h0 = layer_norm(add(x, a0), l0_ln1_gamma, l0_ln1_beta);
        let f0 = add_rows(matmul(gelu(add_rows(matmul(h0, l0_w1), l0_b1)), l0_w2), l0_b2);
        let x0 = layer_norm(add(h0, f0), l0_ln2_gamma, l0_ln2_beta);

        let q1 = add_rows(matmul(x0, l1_wq), l1_bq);
        let k1 = add_rows(matmul(x0, l1_wk), l1_bk);
        let v1 = add_rows(matmul(x0, l1_wv), l1_bv);
        let a1 = add_rows(matmul(attention(q1, k1, v1, key_mask, heads = 2), l1_wo), l1_bo);
        let h1 = layer_norm(add(x0, a1), l1_ln1_gamma, l1_ln1_beta);
        let f1 = add_rows(matmul(gelu(add_rows(matmul(h1, l1_w1), l1_b1)), l1_w2), l1_b2);
        return layer_norm(add(h1, f1), l1_ln2_gamma, l1_ln2_beta);
    }
}

train Fit {
    model = TransformerBlock;
    input x: Tensor<f32, [1024, 128]>;
    input key_mask: Tensor<f32, [8, 128]>;
    target y: Tensor<f32, [1024, 128]>;
    loss = mse(model.forward(x, key_mask), y);
    optimizer = sgd(lr = 0.01);
    steps = 100;
    seed = 42;
    device = vulkan(0);
}
