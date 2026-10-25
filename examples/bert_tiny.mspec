// BERT-tiny (Turc et al. 2019, arXiv 1908.08962): two post-layer-norm encoder layers of
// Devlin et al. 2019 with hidden size 128, 2 attention heads and a feed-forward width of 512,
// learned positions over sequences of 128, in batches of 32, over the 8,192-token WordPiece
// vocabulary of the repository's corpus tokenizer (spec/domains/data.yaml, TokenizerKind).
// Masked-language-model pre-training only, so there is no token-type embedding and no pooler;
// the MLM head's decoder is the token table transposed, with its own bias. 1,486,720
// parameters; Turc et al.'s BERT-tiny has 4.4M at the 30,522 tokens of the original BERT
// release. Dropout 0.1 after the embeddings and each sublayer, and AdamW (lr 1e-4, weight decay
// 0.01, betas 0.9 and 0.999, eps 1e-6) on a linear warm-up and decay. It differs from Devlin et
// al. 2019 in four ways: weight decay applies to every parameter, layer-norm gains and biases
// included; every matrix is initialised xavier-uniform, not from N(0, 0.02); there is no dropout
// on the attention probabilities; and the warm-up is 10 % of the run, not 1 %. The ids,
// positions, key mask, labels and weights are the tensors of a packed `--data` directory,
// declared [B·S] and [B, S]. Its hidden-32 twin,
// crates/modelspec-cli/tests/fixtures/bert_cpu.mspec, is the same program at sizes the CPU
// reference gradient-checks and trains in seconds.
model BertTiny {
    param tok_table: Tensor<f32, [8192, 128]> = xavier_uniform();
    param pos_table: Tensor<f32, [128, 128]> = xavier_uniform();
    param emb_ln_gamma: Tensor<f32, [128]> = ones();
    param emb_ln_beta: Tensor<f32, [128]> = zeros();

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

    param mlm_w: Tensor<f32, [128, 128]> = xavier_uniform();
    param mlm_b: Tensor<f32, [128]> = zeros();
    param mlm_ln_gamma: Tensor<f32, [128]> = ones();
    param mlm_ln_beta: Tensor<f32, [128]> = zeros();
    param mlm_bias: Tensor<f32, [8192]> = zeros();

    fn forward(ids: Tensor<i32, [4096]>, positions: Tensor<i32, [4096]>, key_mask: Tensor<f32, [32, 128]>)
        -> Tensor<f32, [4096, 8192]> {
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
    input ids: Tensor<i32, [4096]>;
    input positions: Tensor<i32, [4096]>;
    input key_mask: Tensor<f32, [32, 128]>;
    target labels: Tensor<i32, [4096]>;
    target weights: Tensor<f32, [4096]>;
    loss = cross_entropy(model.forward(ids, positions, key_mask), labels, weights);
    optimizer = adamw(lr = 1e-4, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-6, warmup_steps = 1000);
    steps = 10000;
    seed = 42;
    device = cpu;
}
