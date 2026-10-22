---
format: aep.planning-md/3
id: story:bert-mlm-cross-entropy
kind: story
status: draft
title: Weighted cross-entropy over integer labels for the MLM loss, gradient-checked on the CPU
summary: Adds cross_entropy(logits, labels i32, weights) and its gradient op on the CPU interpreter, checked by finite differences also through a tied decoder.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-attention
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_cross_entropy.rs
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
  path: crates/modelspec-runtime/tests/gradients_cross_entropy.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated/
- confidence: cited
  path: website/docs/reference/operators.md
revision: 3
---
# Weighted cross-entropy over integer labels for the MLM loss, gradient-checked on the CPU

## Context

`epic:bert-from-scratch` (In scope, Language and operators: "cross-entropy over a vocabulary";
Evaluation: "MLM loss and accuracy on a held-out split"). Only the selected positions of a sequence are predicted, so the
loss is weighted per position: weight 1 at a selected position, 0 elsewhere.
`story:bert-mlm-batches` produces labels and weights in that form.

## What this story decides

- `cross_entropy(logits, labels, weights)`: `logits: Tensor<f32, [N, V]>`,
  `labels: Tensor<i32, [N]>`, `weights: Tensor<f32, [N]>`, result rank 0:
  `sum_n weights[n] * (logsumexp(logits[n]) - logits[n, labels[n]]) / sum_n weights[n]`, the
  log-sum-exp max-subtracted, the sums in ascending `n`. `labels` is the second permitted use of an
  `i32` tensor (the rule of `story:bert-embedding-lookup`). A weight sum of 0, a negative weight or
  a label outside `[0, V)` is a `RuntimeError`, exit 1.
- The gradient with respect to `logits` is an internal op `CrossEntropyGrad(logits, labels,
  weights, g)`, `g * weights[n] * (softmax(logits[n]) - onehot(labels[n])) / sum(weights)`;
  `labels` and `weights` have none.
- The decoder of the MLM head may reuse the token embedding table, `matmul(h, transpose(table))`,
  so one parameter is read twice; the adjoints of both reads add up (DESIGN.md § 4, an adjoint per
  use). This is a parameter read twice, not the `share` of DESIGN.md § 2.3.

## Order on shared files

Fifth of the seven operator stories; the shared files are listed in
`story:bert-embedding-lookup` (Order on shared files). This story depends on
`story:bert-attention` for them; it also edits the `i32` use rule in
`crates/modelspec-core/src/types.rs` that `story:bert-embedding-lookup` introduced.

## Domain relations

None. Operators are the `.mspec` language (DESIGN.md § 2-4; AGENTS.md, Planning and
specification); the ESS edit is the `TensorOperator` variant list, which carries no relation.

## Scope

- `crates/modelspec-core/src/types.rs` (labels as an `i32` use), `ops.rs`, `resolve.rs`,
  `ir/tensor.rs` (`CrossEntropy`, `CrossEntropyGrad`), `ir/verify.rs`, `ir/print.rs`,
  `autodiff.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`.
- `crates/modelspec-runtime/tests/gradients_cross_entropy.rs` (new);
  `crates/modelspec-cli/tests/check_cross_entropy.rs` (new) with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `spec/domains/observatory.yaml` (`TensorOperator`), the committed observatory JSON Schema,
  `crates/modelspec-observatory/`, `ui/src/generated/`.
- `DESIGN.md` § 4 and the revision line; `crates/modelspec-docs/src/operators.rs` and
  `website/docs/reference/operators.md` through `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`cross_entropy` with integer labels and per-position weights checks, lowers, runs on the CPU
interpreter and passes an `f64` finite-difference gradient check, also through a tied decoder.

Scenarios, each run by `task ci` through the command shown:

1. `cd spec && ess specify validate --path .` exits 0 with `CrossEntropy` and `CrossEntropyGrad`
   in `modelspec.observatory.TensorOperator`.
2. `cargo test -p modelspec-runtime --test gradients_cross_entropy`: with `N = 6`, `V = 5` and
   weights `[1, 0, 1, 1, 0, 1]`, the gradient of `cross_entropy(matmul(x, w), labels, weights)`
   with respect to `w` is within `TOL_F64` in `f64` and `TOL_F32` in `f32` of central differences.
3. `cargo test -p modelspec-runtime --test gradients_cross_entropy tied_decoder`: for
   `cross_entropy(matmul(embedding(table, ids), transpose(table)), labels, weights)` the gradient
   with respect to `table` passes the same check, so the adjoints of the two reads are summed.
4. `cargo test -p modelspec-runtime cross_entropy_values`: uniform logits give `ln V` within `1e-6`;
   a row with weight 0 does not change the value when its logits change; logits of magnitude
   `1e3` give a finite loss.
5. `cargo test -p modelspec-runtime cross_entropy_refusals`: all weights 0, a weight of `-1`, and a
   label of `V` each give the `RuntimeError`, not NaN or a panic.
6. `cargo test -p modelspec-cli --test check_cross_entropy`: `f32` labels, labels of length
   `N + 1`, and `logits` of rank 1 each exit 1 with an `E-TYPE` code at the call.
7. `task docs-check` exits 0, and `website/docs/reference/operators.md` lists `cross_entropy`.

## Out of scope

- Label smoothing; an `ignore_index` (a weight of 0 does that).
- Vulkan kernels (`story:bert-vulkan-kernels-attention-loss-adamw`).
