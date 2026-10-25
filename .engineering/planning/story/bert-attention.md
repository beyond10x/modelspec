---
format: aep.planning-md/3
id: story:bert-attention
kind: story
status: active
title: Multi-head bidirectional attention with a key mask, gradient-checked on the CPU
summary: Adds attention(q, k, v, key_mask, heads = A) over flattened [B*S, H] rows, the first named compile-time operator argument, and three gradient ops, with an f64 finite-difference check.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-layer-norm-softmax
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_attention.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/ops.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-observatory/
- confidence: cited
  path: crates/modelspec-observatory/schema/
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients_attention.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated/
- confidence: cited
  path: website/docs/reference/operators.md
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T02:40:31Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-25T02:40:31Z", actor: "human:timo", revision: 5}
---
# Multi-head bidirectional attention with a key mask, gradient-checked on the CPU

## Context

`epic:bert-from-scratch` (In scope, Language and operators: "scaled dot-product multi-head attention
with an attention mask"). DESIGN.md row M3 names "causal and bidirectional masks"; BERT needs only the
bidirectional one, a mask over padding keys. The causal mask belongs to the GPT-like model of M3
and is not drafted here.

The tensors stay rank 2 (M0 has `matmul` and `transpose` on rank 2 only, `TensorOp::MatMul` and
`TensorOp::Transpose`, `crates/modelspec-core/src/ir/tensor.rs:133-137`): a batch of `B`
sequences of length `S` is `N = B * S` rows, sequence `b` being rows `b * S` to `b * S + S - 1`.

## What this story decides

- `attention(q, k, v, key_mask, heads = A)`: `q`, `k`, `v: Tensor<f32, [N, H]>`,
  `key_mask: Tensor<f32, [B, S]>` with `N = B * S`, `H` divisible by `A`, result `[N, H]`. Head
  `a` reads columns `a * H/A` to `(a + 1) * H/A - 1`. Within each sequence and head, the scores are
  `q k^T / sqrt(H/A)`; a key with mask value 0 gets probability exactly 0 and a key with mask value
  1 takes part in a max-subtracted softmax over the unmasked keys; the output is the probabilities
  times `v`, heads concatenated in order. Every query attends to every unmasked key of its own
  sequence (bidirectional). A mask value other than 0 or 1, or a sequence with no unmasked key, is a
  `RuntimeError` of the CPU interpreter, exit 1.
- `heads` is a named argument whose value is an integer literal fixed at compile time; it is the
  first named argument on an operator call (the grammar already has `arg := [IDENT "="] expr`,
  `crates/modelspec-core/src/syntax/parser.rs` module doc). A missing `heads`, a non-literal value,
  or `H` not divisible by it is an `E-TYPE` error; an unknown named argument on any operator is
  refused with a code.
- Gradients: three single-result internal ops `AttentionGradQ`, `AttentionGradK`,
  `AttentionGradV`, each taking `q, k, v, key_mask, g` and recomputing the probabilities; `key_mask`
  has no gradient. Masked keys receive exactly 0 in the gradients for `k` and `v`.

## Order on shared files

Fourth of the seven operator stories; the shared files are listed in
`story:bert-embedding-lookup` (Order on shared files). This story depends on
`story:bert-layer-norm-softmax` for them; it also edits `crates/modelspec-core/src/types.rs` and
`resolve.rs` for named operator arguments, which `story:bert-dropout` reuses.

## Domain relations

None. Operators are the `.mspec` language (DESIGN.md § 2-4; AGENTS.md, Planning and
specification); the ESS edit is the `TensorOperator` variant list, which carries no relation.

## Scope

- `crates/modelspec-core/src/types.rs`, `resolve.rs`, `ops.rs`: named compile-time operator
  arguments, `attention` in `OPERATORS`.
- `crates/modelspec-core/src/ir/tensor.rs` (`Attention { heads }` and the three gradient ops),
  `ir/verify.rs`, `ir/print.rs`, `autodiff.rs`, `lower.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`.
- `crates/modelspec-runtime/tests/gradients_attention.rs` (new);
  `crates/modelspec-cli/tests/check_attention.rs` (new) with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `spec/domains/observatory.yaml` (`TensorOperator`), the committed observatory JSON Schema,
  `crates/modelspec-observatory/`, `ui/src/generated/`.
- `DESIGN.md` § 2.1 (named compile-time operator arguments), § 4 and the revision line;
  `crates/modelspec-docs/src/operators.rs` and `website/docs/reference/operators.md` through
  `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`attention` with two heads and a partial key mask checks, lowers, runs on the CPU interpreter and
passes an `f64` finite-difference gradient check for `q`, `k` and `v`.

Scenarios, each run by `task ci` through the command shown:

1. `cd spec && ess specify validate --path .` exits 0 with `Attention`, `AttentionGradQ`,
   `AttentionGradK` and `AttentionGradV` in `modelspec.observatory.TensorOperator`.
2. `cargo test -p modelspec-runtime --test gradients_attention`: with `B = 2`, `S = 3`, `H = 4`,
   `heads = 2`, `q`, `k`, `v` each `matmul(x, w_q|w_k|w_v)` and a mask that hides the last key of
   sequence 1 only, the gradients of an `mse` loss over the output with respect to `w_q`, `w_k` and
   `w_v` are within `TOL_F64` in `f64` and `TOL_F32` in `f32` of central differences.
3. `cargo test -p modelspec-runtime attention_masked_key`: changing `k` and `v` in the masked row
   leaves the output unchanged bit for bit, and the gradient rows for `k` and `v` at the masked key
   are exactly 0.
4. `cargo test -p modelspec-runtime attention_one_head_reference`: with `heads = 1` and an all-ones
   mask, the output equals `softmax(q k^T / sqrt(H)) v` per sequence computed with the operators of
   `story:bert-layer-norm-softmax` and `matmul`, within `1e-6`.
5. `cargo test -p modelspec-runtime attention_fully_masked`: a sequence whose mask is all 0 gives the
   `RuntimeError`, not NaN.
6. `cargo test -p modelspec-cli --test check_attention`: no `heads`, `heads = 3` with `H = 4`, a
   non-literal `heads`, a `key_mask` whose `B * S` differs from `N`, and an unknown named argument
   each exit 1 with a code at the call.
7. `task docs-check` exits 0, and `website/docs/reference/operators.md` lists `attention`.

## Out of scope

- Causal masking (the GPT-like model of DESIGN.md row M3).
- Rank-3 or rank-4 tensors and general batched `matmul`.
- Vulkan kernels (`story:bert-vulkan-kernels-attention-loss-adamw`).
