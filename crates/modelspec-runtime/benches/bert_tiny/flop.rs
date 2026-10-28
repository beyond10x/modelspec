//! The matmul-only FLOP count of one BERT-tiny pre-training step
//! (story:bert-tiny-vs-pytorch-cuda), a closed form in the batch `B` and the sequence length `S`.
//! Nothing here opens a device; `tests/bert_tiny_flop.rs` includes this file and checks it, and
//! `crates/modelspec-docs/src/benchmarks.rs` checks a recorded `bert-tiny-pretrain` run's
//! `flop_per_step` against [`FLOP_PER_SEQUENCE`].
//!
//! The model is `examples/bert_tiny.mspec`: token and position embeddings, two post-layer-norm
//! encoder layers (hidden `H` = 128, 2 heads, feed-forward `F` = 512), the MLM transform
//! `mlm_w` (`[H, H]`) and the decoder tied to the token table (`[N, H]` by `[H, V]`, vocabulary
//! `V` = 8,192), on `N = B·S` rows, scored on every row.
//!
//! # The FLOP rule, `matmul-only`
//!
//! [`flop_per_step`] counts the matrix products of one training step, forward and backward,
//! `2·m·k·n` for an `[m, k]` by `[k, n]` product, and nothing else: no embedding lookup or its
//! scatter-add, bias, residual, GELU, layer-norm, softmax, mask, dropout, scale, loss or update
//! work. It is the transformer block's rule (`benches/transformer_block/block.rs`,
//! `N·(42·H² + 24·H·F + 24·S·H)`) applied to every product of the model:
//!
//! - Dense ([`dense_flop`]). Each layer has six dense matmuls on `N` rows, `8·N·H² + 4·N·H·F`
//!   forward. Backward each takes the same count again for its weight gradient and again for the
//!   adjoint of its input. Unlike the block, the first layer's input is the embedded batch, whose
//!   tables are trained, so its three projections' input adjoints are counted too:
//!   `3·L·(8·N·H² + 4·N·H·F)`, with `L = 2` layers `48·N·H² + 24·N·H·F`, the block's
//!   `42·N·H²` plus `6·N·H²`.
//! - Head ([`head_flop`]). The MLM transform, `[N, H]` by `[H, H]`, forward, weight gradient and
//!   input adjoint: `6·N·H²`.
//! - Decoder ([`decoder_flop`]). The logits, `[N, H]` by `[H, V]` with the token table
//!   transposed, forward, the table's gradient and the adjoint of its input: `6·N·H·V`.
//! - Attention ([`attention_flop`]). As in the block: per layer the products `Q·Kᵀ` and `P·V`,
//!   `2·N·S·H` each over the heads and sequences, and the four adjoints `dQ`, `dK`, `dP`, `dV` of
//!   the same size, `24·N·S·H` for two layers. ModelSpec's attention kernels recompute the
//!   probabilities in the backward pass; the rule counts the step's mathematics, not a side's
//!   recomputation.
//! - Embeddings ([`embedding_flop`]). A lookup is a gather and its gradient a scatter-add,
//!   neither a product: 0, as the plan's counting rule has it for `Embedding` and
//!   `EmbeddingGrad`.
//!
//! Together `N·(54·H² + 24·H·F + 6·H·V + 24·S·H)`. At `H = 128`, `F = 512`, `V = 8,192`,
//! `S = 128` that is 9,142,272 FLOP per row, 1,170,210,816 per sequence and
//! 37,446,746,112 at `B = 32`, of which the dense, head and decoder parts, 35,836,133,376, are the
//! `MatMul` dispatches of the example's step plan. The opponent
//! `opponents/bert_tiny_pretrain/bert_tiny_pretrain.py` counts the same.

/// The batch the example declares, in sequences.
pub const BATCH: u64 = 32;
/// The sequence length `S`.
pub const SEQ: u64 = 128;
/// The tokens of one step at the example's batch, `B·S`.
pub const TOKENS_PER_STEP: u64 = BATCH * SEQ;
/// The hidden width `H`.
pub const HIDDEN: u64 = 128;
/// The feed-forward width `F`.
pub const FFN: u64 = 512;
/// The vocabulary `V`.
pub const VOCAB: u64 = 8192;
/// The encoder layers `L`.
pub const LAYERS: u64 = 2;
/// The FLOP of one sequence of `S` tokens, `flop_per_step(1, SEQ)`.
pub const FLOP_PER_SEQUENCE: u64 = 1_170_210_816;
/// What the FLOP count counts.
pub const FLOP_RULE: &str = "matmul-only";

/// The dense part, `3·L·(8·N·H² + 4·N·H·F)` with `N = B·S` (module docs).
pub fn dense_flop(batch: u64, seq: u64) -> u64 {
    let n = batch * seq;
    3 * LAYERS * (8 * n * HIDDEN * HIDDEN + 4 * n * HIDDEN * FFN)
}

/// The MLM transform, `6·N·H²` (module docs).
pub fn head_flop(batch: u64, seq: u64) -> u64 {
    6 * batch * seq * HIDDEN * HIDDEN
}

/// The tied decoder, `6·N·H·V` (module docs).
pub fn decoder_flop(batch: u64, seq: u64) -> u64 {
    6 * batch * seq * HIDDEN * VOCAB
}

/// The attention products, `L·6·2·N·S·H` (module docs).
pub fn attention_flop(batch: u64, seq: u64) -> u64 {
    LAYERS * 6 * 2 * batch * seq * seq * HIDDEN
}

/// The embedding lookups and their gradients: no product, 0 (module docs).
pub fn embedding_flop(_batch: u64, _seq: u64) -> u64 {
    0
}

/// The matmul-only FLOP of one training step, `N·(54·H² + 24·H·F + 6·H·V + 24·S·H)`.
pub fn flop_per_step(batch: u64, seq: u64) -> u64 {
    dense_flop(batch, seq)
        + head_flop(batch, seq)
        + decoder_flop(batch, seq)
        + attention_flop(batch, seq)
        + embedding_flop(batch, seq)
}
