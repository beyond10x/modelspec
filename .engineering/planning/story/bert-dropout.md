---
format: aep.planning-md/3
id: story:bert-dropout
kind: story
status: active
title: Dropout drawn from Philox through an explicit per-step RNG token, gradient-checked on the CPU
summary: Adds dropout on Philox stream tag 4 keyed by site and step through an explicit token input, identity outside the training step, with an f64 finite-difference check at a fixed token.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-mlm-cross-entropy
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_dropout.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/
- confidence: cited
  path: crates/modelspec-cli/tests/train_dropout.rs
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/training.rs
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
  path: crates/modelspec-runtime/src/train/init.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/optimizer.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients_dropout.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated/
- confidence: cited
  path: website/docs/reference/operators.md
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T06:47:21Z", actor: "human:timo", revision: 6}
- {from: "proposed", to: "active", at: "2026-10-25T06:47:21Z", actor: "human:timo", revision: 7}
---
# Dropout drawn from Philox through an explicit per-step RNG token, gradient-checked on the CPU

## Context

`epic:bert-from-scratch` (In scope, Language and operators: "dropout with the existing Philox key/counter derivation"). DESIGN.md § 2.3:
"Randomness used in a forward pass (e.g. dropout, post-M2) is an explicit RNG state/token", so that
common-subexpression elimination cannot merge two independent draws. The draw derivation of § 2.3
(key from the seed, counter `[index, stream, id low, id high]`) defines stream tags 0 to 2 only.

## What this story decides

- `dropout(x, p = <float literal>)`, `0 <= p < 1`, any shape. In the training step the result is
  `x * keep / (1 - p)`, element `i` kept when its uniform draw `u_i` (§ 2.3 uniform mapping) is at
  least `p`.
- Stream tag 4 is dropout (coordinator decision D30, 2026-10-25: D26 gave tag 3 to masking and reserved 4 for dropout; `Stream::DataOrder` is 5). A dropout site is named `<Model>.<fn>.dropout#<k>`, `k` the ordinal of
  the call in source order within its function, which does not depend on compiler iteration order.
  At step `s` the draw id of a site is the first 8 bytes of `blake3(site name, then s as 8
  little-endian bytes)`, read as a little-endian `u64`; element `i` takes
  `philox4x32_10([i, 4, id low, id high], key)`.
- The token: each site reads an internal program input, its RNG token, an `i32` tensor of shape
  `[2]` holding that id, which the training host computes every step. The `Dropout` node and its
  internal gradient op `DropoutGrad(token, g)` read the same token, so the backward pass applies the
  same mask without storing it, and two sites never share a node.
- Dropout applies only in the training step. The forward program the runner evaluates on the final
  parameters (the loss at `k = steps`, `crates/modelspec-runtime/src/train/runner.rs` module doc)
  and every evaluation (`story:bert-heldout-evaluation`) lower `dropout` to the identity. DESIGN.md
  § 8 says which printed losses include dropout.
- `p` outside `[0, 1)` or not a literal is an `E-TYPE` error at the call; `p = 0` lowers to the
  identity everywhere.

## Order on shared files

Sixth of the seven operator stories; the shared files are listed in
`story:bert-embedding-lookup` (Order on shared files). This story depends on
`story:bert-mlm-cross-entropy` for them. It also edits `DESIGN.md` § 2.3, which
`story:bert-mlm-batches` edits after it (that story depends on `story:bert-adamw-warmup`, which
depends on this one), and `crates/modelspec-core/src/ir/training.rs`,
`crates/modelspec-runtime/src/train/runner.rs` and `optimizer.rs`, which `story:bert-adamw-warmup`
edits next.

## Domain relations

- The run seed that keys every draw is `modelspec.training.TrainingRun.rng`
  (`modelspec.training.RngState.seed`, the single seed of the run per the summary of `rng` in
  `spec/domains/training.yaml`); the
  algorithm is `modelspec.training.RngAlgorithm` `Philox4x32x10`. No `relations:` entry changes:
  the token is per step and derived, and a checkpoint resumes it from
  `modelspec.checkpoint.Checkpoint.rng` and the step (`spec/domains/checkpoint.yaml`, "RNG state
  at `step`").

## Scope

- `crates/modelspec-core/src/ops.rs`, `resolve.rs`, `types.rs` (the `p` argument),
  `ir/tensor.rs` (`Dropout`, `DropoutGrad`), `ir/verify.rs`, `ir/print.rs`, `autodiff.rs`,
  `lower.rs` (site names, the token inputs, identity outside the training step),
  `ir/training.rs` (the token inputs of the step).
- `crates/modelspec-runtime/src/cpu/interpreter.rs`; `src/train/init.rs` (stream tag 4, already declared as `Stream::Dropout` by story:bert-mlm-batches if present, else added),
  `runner.rs` and `optimizer.rs` (the host computes the tokens of each step).
- `crates/modelspec-runtime/tests/gradients_dropout.rs` (new);
  `crates/modelspec-cli/tests/check_dropout.rs` and `train_dropout.rs` (new) with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `spec/domains/observatory.yaml` (`TensorOperator`), the committed observatory JSON Schema,
  `crates/modelspec-observatory/`, `ui/src/generated/`.
- `DESIGN.md` § 2.3 (tag 4 and the token), § 4, § 8 and the revision line;
  `crates/modelspec-docs/src/operators.rs` and `website/docs/reference/operators.md` through
  `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`dropout` draws a reproducible mask per site and per step from Philox stream 3 through an explicit
token, passes an `f64` finite-difference gradient check at a fixed token, and is the identity
outside the training step.

Scenarios, each run by `task ci` through the command shown:

1. `cd spec && ess specify validate --path .` exits 0 with `Dropout` and `DropoutGrad` in
   `modelspec.observatory.TensorOperator`, and
   `grep -Eq "^\*\*Revision:\*\* [0-9.]+ \([0-9-]+\): [^;]*dropout" DESIGN.md` exits 0.
2. `cargo test -p modelspec-runtime --test gradients_dropout`: at a fixed token and `p = 0.3`, the
   gradient of an `mse` loss over `dropout(matmul(x, w), p = 0.3)` with respect to `w` is within
   `TOL_F64` in `f64` and `TOL_F32` in `f32` of central differences.
3. `cargo test -p modelspec-runtime dropout_mask_derivation`: for a fixed seed, site name and step,
   the kept elements equal those the test derives from `philox4x32_10` with counter
   `[i, 4, id low, id high]`, and the kept share over `10_000` elements at `p = 0.1` lies within
   `0.9 +- 0.01`.
4. `cargo test -p modelspec-runtime dropout_sites_and_steps_differ`: two dropout calls on the same
   value in one function give different masks, both nodes stay in the Tensor IR after DCE, and one
   site gives different masks at steps 0 and 1.
5. `cargo test -p modelspec-cli --test train_dropout`: `modelspec train --backend cpu` on a
   fixture with dropout, run twice with one seed, prints identical stdout; with `p = 0` it prints
   the stdout of the same fixture without the call; the final loss line equals the loss of the
   forward program without dropout on the printed final parameters.
6. `cargo test -p modelspec-cli --test check_dropout`: `p = 1.0`, `p = -0.1` and a non-literal `p`
   exit 1 with an `E-TYPE` code at the call.
7. `task docs-check` exits 0, and `website/docs/reference/operators.md` lists `dropout`.

## Out of scope

- Dropout on the GPU (`story:bert-vulkan-kernels-rowwise`).
- Attention-probability dropout inside `attention`: BERT applies it, the fused operator of
  `story:bert-attention` has none, and the epic does not ask for it.
