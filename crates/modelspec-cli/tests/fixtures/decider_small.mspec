// decider_small.mspec: the decider's JEPA decision model in its 0.21.0 workaround form, the gap
// scout's decider-current draft of 2026-10-28 without stop_gradient and sigreg (story
// decider-dense-named-data), at sizes small enough for a test. Hidden H = 8, 2 heads,
// feed-forward 16. Event encoder E: 4 post-LN layers over T = 4 tokens per event, one pool of
// U = 4 events per step (16 tokens). Context encoder C: 2 layers over the k = 2 events of each
// of D = 2 decisions. Predictor P: 2-layer MLP, hidden 16. Action table: 3 templates plus one
// abstain row (A1 = 4). Vocabulary 32; 4 session-position buckets.
//
// The losses are the decider's: smooth-L1 of the predicted next event, a stand-in mean(s * s)
// where SIGReg goes, and cross_entropy over the candidates and the abstain logit. Every
// workaround of the draft stays: hand-inlined layers, selection matmuls for the reshape, a split
// predictor weight for the concatenation, smooth-L1 from relu, sub(0.0, x) for a sign, an
// additive candidate bias and the shape-only index tensors shipped as data. The train block
// reads 19 named tensors a step, 12 of them i32.
model Decider {
    param tok_table: Tensor<f32, [32, 8]> = xavier_uniform();
    param tok_pos_table: Tensor<f32, [4, 8]> = xavier_uniform();
    param e_ln_gamma: Tensor<f32, [8]> = ones();
    param e_ln_beta: Tensor<f32, [8]> = zeros();

    param e0_wq: Tensor<f32, [8, 8]> = xavier_uniform();
    param e0_bq: Tensor<f32, [8]> = zeros();
    param e0_wk: Tensor<f32, [8, 8]> = xavier_uniform();
    param e0_bk: Tensor<f32, [8]> = zeros();
    param e0_wv: Tensor<f32, [8, 8]> = xavier_uniform();
    param e0_bv: Tensor<f32, [8]> = zeros();
    param e0_wo: Tensor<f32, [8, 8]> = xavier_uniform();
    param e0_bo: Tensor<f32, [8]> = zeros();
    param e0_ln1_gamma: Tensor<f32, [8]> = ones();
    param e0_ln1_beta: Tensor<f32, [8]> = zeros();
    param e0_w1: Tensor<f32, [8, 16]> = xavier_uniform();
    param e0_b1: Tensor<f32, [16]> = zeros();
    param e0_w2: Tensor<f32, [16, 8]> = xavier_uniform();
    param e0_b2: Tensor<f32, [8]> = zeros();
    param e0_ln2_gamma: Tensor<f32, [8]> = ones();
    param e0_ln2_beta: Tensor<f32, [8]> = zeros();

    param e1_wq: Tensor<f32, [8, 8]> = xavier_uniform();
    param e1_bq: Tensor<f32, [8]> = zeros();
    param e1_wk: Tensor<f32, [8, 8]> = xavier_uniform();
    param e1_bk: Tensor<f32, [8]> = zeros();
    param e1_wv: Tensor<f32, [8, 8]> = xavier_uniform();
    param e1_bv: Tensor<f32, [8]> = zeros();
    param e1_wo: Tensor<f32, [8, 8]> = xavier_uniform();
    param e1_bo: Tensor<f32, [8]> = zeros();
    param e1_ln1_gamma: Tensor<f32, [8]> = ones();
    param e1_ln1_beta: Tensor<f32, [8]> = zeros();
    param e1_w1: Tensor<f32, [8, 16]> = xavier_uniform();
    param e1_b1: Tensor<f32, [16]> = zeros();
    param e1_w2: Tensor<f32, [16, 8]> = xavier_uniform();
    param e1_b2: Tensor<f32, [8]> = zeros();
    param e1_ln2_gamma: Tensor<f32, [8]> = ones();
    param e1_ln2_beta: Tensor<f32, [8]> = zeros();

    param e2_wq: Tensor<f32, [8, 8]> = xavier_uniform();
    param e2_bq: Tensor<f32, [8]> = zeros();
    param e2_wk: Tensor<f32, [8, 8]> = xavier_uniform();
    param e2_bk: Tensor<f32, [8]> = zeros();
    param e2_wv: Tensor<f32, [8, 8]> = xavier_uniform();
    param e2_bv: Tensor<f32, [8]> = zeros();
    param e2_wo: Tensor<f32, [8, 8]> = xavier_uniform();
    param e2_bo: Tensor<f32, [8]> = zeros();
    param e2_ln1_gamma: Tensor<f32, [8]> = ones();
    param e2_ln1_beta: Tensor<f32, [8]> = zeros();
    param e2_w1: Tensor<f32, [8, 16]> = xavier_uniform();
    param e2_b1: Tensor<f32, [16]> = zeros();
    param e2_w2: Tensor<f32, [16, 8]> = xavier_uniform();
    param e2_b2: Tensor<f32, [8]> = zeros();
    param e2_ln2_gamma: Tensor<f32, [8]> = ones();
    param e2_ln2_beta: Tensor<f32, [8]> = zeros();

    param e3_wq: Tensor<f32, [8, 8]> = xavier_uniform();
    param e3_bq: Tensor<f32, [8]> = zeros();
    param e3_wk: Tensor<f32, [8, 8]> = xavier_uniform();
    param e3_bk: Tensor<f32, [8]> = zeros();
    param e3_wv: Tensor<f32, [8, 8]> = xavier_uniform();
    param e3_bv: Tensor<f32, [8]> = zeros();
    param e3_wo: Tensor<f32, [8, 8]> = xavier_uniform();
    param e3_bo: Tensor<f32, [8]> = zeros();
    param e3_ln1_gamma: Tensor<f32, [8]> = ones();
    param e3_ln1_beta: Tensor<f32, [8]> = zeros();
    param e3_w1: Tensor<f32, [8, 16]> = xavier_uniform();
    param e3_b1: Tensor<f32, [16]> = zeros();
    param e3_w2: Tensor<f32, [16, 8]> = xavier_uniform();
    param e3_b2: Tensor<f32, [8]> = zeros();
    param e3_ln2_gamma: Tensor<f32, [8]> = ones();
    param e3_ln2_beta: Tensor<f32, [8]> = zeros();

    param slot_table: Tensor<f32, [2, 8]> = xavier_uniform();
    param sess_pos_table: Tensor<f32, [4, 8]> = xavier_uniform();
    param c_ln_gamma: Tensor<f32, [8]> = ones();
    param c_ln_beta: Tensor<f32, [8]> = zeros();

    param c0_wq: Tensor<f32, [8, 8]> = xavier_uniform();
    param c0_bq: Tensor<f32, [8]> = zeros();
    param c0_wk: Tensor<f32, [8, 8]> = xavier_uniform();
    param c0_bk: Tensor<f32, [8]> = zeros();
    param c0_wv: Tensor<f32, [8, 8]> = xavier_uniform();
    param c0_bv: Tensor<f32, [8]> = zeros();
    param c0_wo: Tensor<f32, [8, 8]> = xavier_uniform();
    param c0_bo: Tensor<f32, [8]> = zeros();
    param c0_ln1_gamma: Tensor<f32, [8]> = ones();
    param c0_ln1_beta: Tensor<f32, [8]> = zeros();
    param c0_w1: Tensor<f32, [8, 16]> = xavier_uniform();
    param c0_b1: Tensor<f32, [16]> = zeros();
    param c0_w2: Tensor<f32, [16, 8]> = xavier_uniform();
    param c0_b2: Tensor<f32, [8]> = zeros();
    param c0_ln2_gamma: Tensor<f32, [8]> = ones();
    param c0_ln2_beta: Tensor<f32, [8]> = zeros();

    param c1_wq: Tensor<f32, [8, 8]> = xavier_uniform();
    param c1_bq: Tensor<f32, [8]> = zeros();
    param c1_wk: Tensor<f32, [8, 8]> = xavier_uniform();
    param c1_bk: Tensor<f32, [8]> = zeros();
    param c1_wv: Tensor<f32, [8, 8]> = xavier_uniform();
    param c1_bv: Tensor<f32, [8]> = zeros();
    param c1_wo: Tensor<f32, [8, 8]> = xavier_uniform();
    param c1_bo: Tensor<f32, [8]> = zeros();
    param c1_ln1_gamma: Tensor<f32, [8]> = ones();
    param c1_ln1_beta: Tensor<f32, [8]> = zeros();
    param c1_w1: Tensor<f32, [8, 16]> = xavier_uniform();
    param c1_b1: Tensor<f32, [16]> = zeros();
    param c1_w2: Tensor<f32, [16, 8]> = xavier_uniform();
    param c1_b2: Tensor<f32, [8]> = zeros();
    param c1_ln2_gamma: Tensor<f32, [8]> = ones();
    param c1_ln2_beta: Tensor<f32, [8]> = zeros();

    param act_table: Tensor<f32, [4, 8]> = xavier_uniform();

    param p_ws: Tensor<f32, [8, 16]> = xavier_uniform();
    param p_wa: Tensor<f32, [8, 16]> = xavier_uniform();
    param p_b1: Tensor<f32, [16]> = zeros();
    param p_w2: Tensor<f32, [16, 8]> = xavier_uniform();
    param p_b2: Tensor<f32, [8]> = zeros();

    param h_ws: Tensor<f32, [8, 1]> = xavier_uniform();
    param h_wh: Tensor<f32, [8, 1]> = xavier_uniform();
    param h_b: Tensor<f32, [1]> = zeros();

    fn forward(
        tok_ids: Tensor<i32, [16]>,
        tok_pos: Tensor<i32, [16]>,
        tok_mask: Tensor<f32, [4, 4]>,
        evt_row: Tensor<i32, [4]>,
        ctx_idx: Tensor<i32, [4]>,
        ctx_slot: Tensor<i32, [4]>,
        ctx_sess: Tensor<i32, [4]>,
        ctx_mask: Tensor<f32, [2, 2]>,
        last_row: Tensor<i32, [2]>,
        next_evt: Tensor<i32, [2]>,
        pair_ctx: Tensor<i32, [8]>,
        pair_act: Tensor<i32, [8]>,
        taken_pair: Tensor<i32, [2]>,
        fold_rows: Tensor<f32, [2, 8]>,
        fold_cols: Tensor<f32, [8, 4]>,
        fold_bcast: Tensor<f32, [1, 4]>,
        cand_bias: Tensor<f32, [2, 4]>,
        labels: Tensor<i32, [2]>,
        label_weights: Tensor<f32, [2]>)
        -> Tensor<f32, []> {
        let t0 = layer_norm(add(embedding(tok_table, tok_ids), embedding(tok_pos_table, tok_pos)), e_ln_gamma, e_ln_beta);

        let eq0 = add_rows(matmul(t0, e0_wq), e0_bq);
        let ek0 = add_rows(matmul(t0, e0_wk), e0_bk);
        let ev0 = add_rows(matmul(t0, e0_wv), e0_bv);
        let ea0 = add_rows(matmul(attention(eq0, ek0, ev0, tok_mask, heads = 2), e0_wo), e0_bo);
        let eh0 = layer_norm(add(t0, ea0), e0_ln1_gamma, e0_ln1_beta);
        let ef0 = add_rows(matmul(gelu(add_rows(matmul(eh0, e0_w1), e0_b1)), e0_w2), e0_b2);
        let t1 = layer_norm(add(eh0, ef0), e0_ln2_gamma, e0_ln2_beta);

        let eq1 = add_rows(matmul(t1, e1_wq), e1_bq);
        let ek1 = add_rows(matmul(t1, e1_wk), e1_bk);
        let ev1 = add_rows(matmul(t1, e1_wv), e1_bv);
        let ea1 = add_rows(matmul(attention(eq1, ek1, ev1, tok_mask, heads = 2), e1_wo), e1_bo);
        let eh1 = layer_norm(add(t1, ea1), e1_ln1_gamma, e1_ln1_beta);
        let ef1 = add_rows(matmul(gelu(add_rows(matmul(eh1, e1_w1), e1_b1)), e1_w2), e1_b2);
        let t2 = layer_norm(add(eh1, ef1), e1_ln2_gamma, e1_ln2_beta);

        let eq2 = add_rows(matmul(t2, e2_wq), e2_bq);
        let ek2 = add_rows(matmul(t2, e2_wk), e2_bk);
        let ev2 = add_rows(matmul(t2, e2_wv), e2_bv);
        let ea2 = add_rows(matmul(attention(eq2, ek2, ev2, tok_mask, heads = 2), e2_wo), e2_bo);
        let eh2 = layer_norm(add(t2, ea2), e2_ln1_gamma, e2_ln1_beta);
        let ef2 = add_rows(matmul(gelu(add_rows(matmul(eh2, e2_w1), e2_b1)), e2_w2), e2_b2);
        let t3 = layer_norm(add(eh2, ef2), e2_ln2_gamma, e2_ln2_beta);

        let eq3 = add_rows(matmul(t3, e3_wq), e3_bq);
        let ek3 = add_rows(matmul(t3, e3_wk), e3_bk);
        let ev3 = add_rows(matmul(t3, e3_wv), e3_bv);
        let ea3 = add_rows(matmul(attention(eq3, ek3, ev3, tok_mask, heads = 2), e3_wo), e3_bo);
        let eh3 = layer_norm(add(t3, ea3), e3_ln1_gamma, e3_ln1_beta);
        let ef3 = add_rows(matmul(gelu(add_rows(matmul(eh3, e3_w1), e3_b1)), e3_w2), e3_b2);
        let t4 = layer_norm(add(eh3, ef3), e3_ln2_gamma, e3_ln2_beta);

        let events = embedding(t4, evt_row);

        let c0 = layer_norm(add(add(embedding(events, ctx_idx), embedding(slot_table, ctx_slot)), embedding(sess_pos_table, ctx_sess)), c_ln_gamma, c_ln_beta);

        let cq0 = add_rows(matmul(c0, c0_wq), c0_bq);
        let ck0 = add_rows(matmul(c0, c0_wk), c0_bk);
        let cv0 = add_rows(matmul(c0, c0_wv), c0_bv);
        let ca0 = add_rows(matmul(attention(cq0, ck0, cv0, ctx_mask, heads = 2), c0_wo), c0_bo);
        let ch0 = layer_norm(add(c0, ca0), c0_ln1_gamma, c0_ln1_beta);
        let cf0 = add_rows(matmul(gelu(add_rows(matmul(ch0, c0_w1), c0_b1)), c0_w2), c0_b2);
        let c1 = layer_norm(add(ch0, cf0), c0_ln2_gamma, c0_ln2_beta);

        let cq1 = add_rows(matmul(c1, c1_wq), c1_bq);
        let ck1 = add_rows(matmul(c1, c1_wk), c1_bk);
        let cv1 = add_rows(matmul(c1, c1_wv), c1_bv);
        let ca1 = add_rows(matmul(attention(cq1, ck1, cv1, ctx_mask, heads = 2), c1_wo), c1_bo);
        let ch1 = layer_norm(add(c1, ca1), c1_ln1_gamma, c1_ln1_beta);
        let cf1 = add_rows(matmul(gelu(add_rows(matmul(ch1, c1_w1), c1_b1)), c1_w2), c1_b2);
        let c2 = layer_norm(add(ch1, cf1), c1_ln2_gamma, c1_ln2_beta);

        let s = embedding(c2, last_row);

        let s_next = embedding(events, next_evt);

        let ps = matmul(s, p_ws);
        let pa = matmul(act_table, p_wa);
        let ph = gelu(add_rows(add(embedding(ps, pair_ctx), embedding(pa, pair_act)), p_b1));
        let s_hat = add_rows(matmul(ph, p_w2), p_b2);

        let s_hat_taken = embedding(s_hat, taken_pair);
        let d = sub(s_hat_taken, s_next);
        let abs_d = add(relu(d), relu(sub(0.0, d)));
        let clip_d = sub(abs_d, relu(sub(abs_d, 1.0)));
        let l_jepa = mean(add(mul(mul(clip_d, clip_d), 0.5), sub(abs_d, clip_d)));

        let l_sig = mean(mul(s, s));

        let score = add_rows(add(embedding(matmul(s, h_ws), pair_ctx), matmul(s_hat, h_wh)), h_b);
        let logits = add(matmul(fold_rows, mul(matmul(score, fold_bcast), fold_cols)), cand_bias);
        let l_dec = cross_entropy(logits, labels, label_weights);

        return add(add(l_jepa, mul(l_sig, 0.05)), l_dec);
    }
}

train Joint {
    model = Decider;
    input tok_ids: Tensor<i32, [16]>;
    input tok_pos: Tensor<i32, [16]>;
    input tok_mask: Tensor<f32, [4, 4]>;
    input evt_row: Tensor<i32, [4]>;
    input ctx_idx: Tensor<i32, [4]>;
    input ctx_slot: Tensor<i32, [4]>;
    input ctx_sess: Tensor<i32, [4]>;
    input ctx_mask: Tensor<f32, [2, 2]>;
    input last_row: Tensor<i32, [2]>;
    input next_evt: Tensor<i32, [2]>;
    input pair_ctx: Tensor<i32, [8]>;
    input pair_act: Tensor<i32, [8]>;
    input taken_pair: Tensor<i32, [2]>;
    input fold_rows: Tensor<f32, [2, 8]>;
    input fold_cols: Tensor<f32, [8, 4]>;
    input fold_bcast: Tensor<f32, [1, 4]>;
    input cand_bias: Tensor<f32, [2, 4]>;
    target labels: Tensor<i32, [2]>;
    target label_weights: Tensor<f32, [2]>;
    loss = model.forward(tok_ids, tok_pos, tok_mask, evt_row, ctx_idx, ctx_slot, ctx_sess, ctx_mask, last_row, next_evt, pair_ctx, pair_act, taken_pair, fold_rows, fold_cols, fold_bcast, cand_bias, labels, label_weights);
    optimizer = adamw(lr = 1e-3, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-6, warmup_steps = 2);
    steps = 10;
    seed = 1;
    device = cpu;
}
