// The hidden-32 twin of examples/bert_tiny.mspec: the same program with H = 32, 2 heads,
// F = 64, S = 16, batches of 2 and the 300-token vocabulary of the fixture tokenizer, 28,684
// parameters. tests/gradients_bert_cpu.rs of modelspec-runtime checks every parameter's gradient
// against central differences in f64, and tests/bert_cpu_train.rs trains it twice from one seed
// on the packed fixture corpus.
model BertTiny {
    param tok_table: Tensor<f32, [300, 32]> = xavier_uniform();
    param pos_table: Tensor<f32, [16, 32]> = xavier_uniform();
    param emb_ln_gamma: Tensor<f32, [32]> = ones();
    param emb_ln_beta: Tensor<f32, [32]> = zeros();

    param l0_wq: Tensor<f32, [32, 32]> = xavier_uniform();
    param l0_bq: Tensor<f32, [32]> = zeros();
    param l0_wk: Tensor<f32, [32, 32]> = xavier_uniform();
    param l0_bk: Tensor<f32, [32]> = zeros();
    param l0_wv: Tensor<f32, [32, 32]> = xavier_uniform();
    param l0_bv: Tensor<f32, [32]> = zeros();
    param l0_wo: Tensor<f32, [32, 32]> = xavier_uniform();
    param l0_bo: Tensor<f32, [32]> = zeros();
    param l0_ln1_gamma: Tensor<f32, [32]> = ones();
    param l0_ln1_beta: Tensor<f32, [32]> = zeros();
    param l0_w1: Tensor<f32, [32, 64]> = xavier_uniform();
    param l0_b1: Tensor<f32, [64]> = zeros();
    param l0_w2: Tensor<f32, [64, 32]> = xavier_uniform();
    param l0_b2: Tensor<f32, [32]> = zeros();
    param l0_ln2_gamma: Tensor<f32, [32]> = ones();
    param l0_ln2_beta: Tensor<f32, [32]> = zeros();

    param l1_wq: Tensor<f32, [32, 32]> = xavier_uniform();
    param l1_bq: Tensor<f32, [32]> = zeros();
    param l1_wk: Tensor<f32, [32, 32]> = xavier_uniform();
    param l1_bk: Tensor<f32, [32]> = zeros();
    param l1_wv: Tensor<f32, [32, 32]> = xavier_uniform();
    param l1_bv: Tensor<f32, [32]> = zeros();
    param l1_wo: Tensor<f32, [32, 32]> = xavier_uniform();
    param l1_bo: Tensor<f32, [32]> = zeros();
    param l1_ln1_gamma: Tensor<f32, [32]> = ones();
    param l1_ln1_beta: Tensor<f32, [32]> = zeros();
    param l1_w1: Tensor<f32, [32, 64]> = xavier_uniform();
    param l1_b1: Tensor<f32, [64]> = zeros();
    param l1_w2: Tensor<f32, [64, 32]> = xavier_uniform();
    param l1_b2: Tensor<f32, [32]> = zeros();
    param l1_ln2_gamma: Tensor<f32, [32]> = ones();
    param l1_ln2_beta: Tensor<f32, [32]> = zeros();

    param mlm_w: Tensor<f32, [32, 32]> = xavier_uniform();
    param mlm_b: Tensor<f32, [32]> = zeros();
    param mlm_ln_gamma: Tensor<f32, [32]> = ones();
    param mlm_ln_beta: Tensor<f32, [32]> = zeros();
    param mlm_bias: Tensor<f32, [300]> = zeros();

    fn forward(ids: Tensor<i32, [32]>, positions: Tensor<i32, [32]>, key_mask: Tensor<f32, [2, 16]>)
        -> Tensor<f32, [32, 300]> {
        let e = dropout(layer_norm(add(embedding(tok_table, ids), embedding(pos_table, positions)), emb_ln_gamma, emb_ln_beta), p = 0.1);

        let q0 = add_rows(matmul(e, l0_wq), l0_bq);
        let k0 = add_rows(matmul(e, l0_wk), l0_bk);
        let v0 = add_rows(matmul(e, l0_wv), l0_bv);
        let a0 = add_rows(matmul(attention(q0, k0, v0, key_mask, heads = 2), l0_wo), l0_bo);
        let h0 = layer_norm(add(e, dropout(a0, p = 0.1)), l0_ln1_gamma, l0_ln1_beta);
        let f0 = add_rows(matmul(gelu(add_rows(matmul(h0, l0_w1), l0_b1)), l0_w2), l0_b2);
        let x0 = layer_norm(add(h0, dropout(f0, p = 0.1)), l0_ln2_gamma, l0_ln2_beta);

        let q1 = add_rows(matmul(x0, l1_wq), l1_bq);
        let k1 = add_rows(matmul(x0, l1_wk), l1_bk);
        let v1 = add_rows(matmul(x0, l1_wv), l1_bv);
        let a1 = add_rows(matmul(attention(q1, k1, v1, key_mask, heads = 2), l1_wo), l1_bo);
        let h1 = layer_norm(add(x0, dropout(a1, p = 0.1)), l1_ln1_gamma, l1_ln1_beta);
        let f1 = add_rows(matmul(gelu(add_rows(matmul(h1, l1_w1), l1_b1)), l1_w2), l1_b2);
        let x1 = layer_norm(add(h1, dropout(f1, p = 0.1)), l1_ln2_gamma, l1_ln2_beta);

        let m = layer_norm(gelu(add_rows(matmul(x1, mlm_w), mlm_b)), mlm_ln_gamma, mlm_ln_beta);
        return add_rows(matmul(m, transpose(tok_table)), mlm_bias);
    }
}

train Pretrain {
    model = BertTiny;
    input ids: Tensor<i32, [32]>;
    input positions: Tensor<i32, [32]>;
    input key_mask: Tensor<f32, [2, 16]>;
    target labels: Tensor<i32, [32]>;
    target weights: Tensor<f32, [32]>;
    loss = cross_entropy(model.forward(ids, positions, key_mask), labels, weights);
    optimizer = adamw(lr = 1e-4, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-6, warmup_steps = 50);
    steps = 500;
    seed = 42;
    device = cpu;
}
