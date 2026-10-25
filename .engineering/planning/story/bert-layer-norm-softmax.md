---
format: aep.planning-md/3
id: story:bert-layer-norm-softmax
kind: story
status: implemented
title: Layer norm and softmax over the last axis, gradient-checked on the CPU
summary: Adds layer_norm (eps 1e-12) and a max-subtracted softmax with their internal gradient ops on the CPU interpreter, each with an f64 finite-difference check.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-dense-gelu
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_norm.rs
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
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-observatory/
- confidence: cited
  path: crates/modelspec-observatory/schema/
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients_norm.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated/
- confidence: cited
  path: website/docs/reference/operators.md
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T00:34:50Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-25T00:34:50Z", actor: "human:timo", revision: 5}
- {from: "active", to: "implemented", at: "2026-10-25T02:29:32Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# Layer norm and softmax over the last axis, gradient-checked on the CPU

## Context

`epic:bert-from-scratch` (In scope, Language and operators: "layer norm, softmax"). Both normalize along the
last axis of a `[N, H]` tensor, so they share the row-wise reduction pattern; the Vulkan kernels
for both are `story:bert-vulkan-kernels-rowwise`.

## What this story decides

- `layer_norm(x, gamma, beta)`: `x: Tensor<f32, [N, H]>`, `gamma` and `beta` of shape `[H]`,
  result `[N, H]`: per row, `(x - mean) / sqrt(var + eps) * gamma + beta`, `var` the biased
  variance, `eps = 1e-12` fixed (the value BERT uses; Devlin et al. 2019 reference
  implementation). The interpreter computes mean and variance in the element type in ascending
  column order. Gradients: internal ops `LayerNormGradX(x, gamma, g) -> [N, H]` and
  `LayerNormGradGamma(x, g) -> [H]`; the gradient for `beta` is `SumRows(g)` from
  `story:bert-dense-gelu`.
- `softmax(x)`: `x: Tensor<f32, [N, K]>`, per row `exp(x - max) / sum(exp(x - max))`. Gradient: an
  internal op `SoftmaxGrad(y, g) = y * (g - rowsum(g * y))` with `y` the forward result.
- Rank other than 2, or `gamma` or `beta` not of shape `[H]`, is an `E-TYPE` error at the call.

## Order on shared files

Third of the seven operator stories; the shared files are listed in
`story:bert-embedding-lookup` (Order on shared files). This story depends on
`story:bert-dense-gelu` for them and for `SumRows`.

## Domain relations

None. Operators are the `.mspec` language (DESIGN.md § 2-4; AGENTS.md, Planning and
specification); the ESS edit is the `TensorOperator` variant list, which carries no relation.

## Scope

- `crates/modelspec-core/src/ops.rs`, `resolve.rs`, `ir/tensor.rs` (`LayerNorm`,
  `LayerNormGradX`, `LayerNormGradGamma`, `Softmax`, `SoftmaxGrad`), `ir/verify.rs`,
  `ir/print.rs`, `autodiff.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`.
- `crates/modelspec-runtime/tests/gradients_norm.rs` (new);
  `crates/modelspec-cli/tests/check_norm.rs` (new) with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `spec/domains/observatory.yaml` (`TensorOperator`), the committed observatory JSON Schema,
  `crates/modelspec-observatory/`, `ui/src/generated/`.
- `DESIGN.md` § 4 and the revision line; `crates/modelspec-docs/src/operators.rs` and
  `website/docs/reference/operators.md` through `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`layer_norm` and `softmax` check, lower, run on the CPU interpreter and pass an `f64`
finite-difference gradient check each.

Scenarios, each run by `task ci` through the command shown:

1. `cd spec && ess specify validate --path .` exits 0 with the five new variants in
   `modelspec.observatory.TensorOperator`.
2. `cargo test -p modelspec-runtime --test gradients_norm layer_norm`: the gradients of an `mse`
   loss over `layer_norm(matmul(x, w), gamma, beta)` with respect to `w`, `gamma` and `beta`,
   `N = 4`, `H = 5`, are within `TOL_F64` in `f64` and `TOL_F32` in `f32` of central differences.
3. `cargo test -p modelspec-runtime --test gradients_norm softmax`: the gradient of an `mse` loss
   over `softmax(matmul(x, w))` with respect to `w`, `K = 6`, passes the same check.
4. `cargo test -p modelspec-runtime softmax_large_logits`: `softmax` of a row `[1000, 999, 0]` in
   `f32` is finite, sums to 1 within `1e-6`, and equals the `f64` result within `1e-6`.
5. `cargo test -p modelspec-runtime layer_norm_rows`: every output row of `layer_norm` with
   `gamma = 1`, `beta = 0` has mean within `1e-6` of 0 and biased variance within `1e-5` of 1.
6. `cargo test -p modelspec-cli --test check_norm`: `layer_norm` with a `gamma` of shape `[H + 1]`
   and `softmax` of a rank-1 tensor exit 1 with an `E-TYPE` code at the call.
7. `task docs-check` exits 0, and `website/docs/reference/operators.md` lists `layer_norm` and
   `softmax`.

## Out of scope

- A configurable `eps` or axis.
- Vulkan kernels (`story:bert-vulkan-kernels-rowwise`).
