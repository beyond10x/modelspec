---
format: aep.planning-md/3
id: story:bert-dense-gelu
kind: story
status: implemented
title: Dense layers with a bias row and GELU, gradient-checked on the CPU
summary: Adds add_rows with its SumRows gradient and the tanh-form gelu on the CPU interpreter, each with an f64 finite-difference check.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-embedding-lookup
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_dense.rs
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
  path: crates/modelspec-runtime/tests/gradients_dense.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated/
- confidence: cited
  path: website/docs/reference/operators.md
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T22:25:57Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-24T22:25:57Z", actor: "human:timo", revision: 5}
- {from: "active", to: "implemented", at: "2026-10-25T00:09:35Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# Dense layers with a bias row and GELU, gradient-checked on the CPU

## Context

`epic:bert-from-scratch` (In scope, Language and operators: "GELU"). A BERT dense layer is `x W + b` with `x` of
shape `[N, H]` and `b` of shape `[H]`. DESIGN.md § 2.2 allows no implicit non-scalar broadcasting
and the M0 `add` takes the same shapes or one rank-0 operand (`TensorOp::Add`,
`crates/modelspec-core/src/ir/tensor.rs:129`), so the bias needs an explicit operator.

## What this story decides

- `add_rows(x, b)`: `x: Tensor<f32, [N, H]>`, `b: Tensor<f32, [H]>`, result `[N, H]`, `b` added to
  every row. The gradient is the adjoint for `x` and an internal gradient op `SumRows` (the sum
  over the row axis, `[N, H] -> [H]`, rows summed in ascending order) for `b`.
- `gelu(x)`: elementwise, the tanh form `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` (Hendrycks
  and Gimpel 2016, the form the original BERT release uses), on any shape. Its gradient is an
  internal op `GeluGrad(x, g)`, the analytic derivative of that form times `g`. DESIGN.md § 4
  records which form, so the Vulkan kernel computes the same function.

## Order on shared files

Second of the seven operator stories; the shared files are listed in
`story:bert-embedding-lookup` (Order on shared files). This story depends on
`story:bert-embedding-lookup` for all of them, and also for `crates/modelspec-runtime/tests/common/mod.rs`,
which that story creates.

## Domain relations

None. Operators are the `.mspec` language, held by DESIGN.md § 2-4 and not by an ESS domain
(AGENTS.md, Planning and specification); the only ESS edit is two `TensorOperator` variant lists
in `spec/domains/observatory.yaml`, which carry no relation.

## Scope

- `crates/modelspec-core/src/ops.rs`, `resolve.rs`, `ir/tensor.rs` (`AddRows`, `SumRows`, `Gelu`,
  `GeluGrad`), `ir/verify.rs`, `ir/print.rs`, `autodiff.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`.
- `crates/modelspec-runtime/tests/gradients_dense.rs` (new); `crates/modelspec-cli/tests/check_dense.rs`
  (new) with its fixtures under `crates/modelspec-cli/tests/fixtures/`.
- `spec/domains/observatory.yaml` (`TensorOperator`), the committed observatory JSON Schema,
  `crates/modelspec-observatory/`, `ui/src/generated/`.
- `DESIGN.md` § 4 and the revision line; `crates/modelspec-docs/src/operators.rs` and
  `website/docs/reference/operators.md` through `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`add_rows` and `gelu` check, lower, run on the CPU interpreter and pass an `f64` finite-difference
gradient check each.

Scenarios, each run by `task ci` through the command shown:

1. `cd spec && ess specify validate --path .` exits 0 with `AddRows`, `SumRows`, `Gelu` and
   `GeluGrad` in `modelspec.observatory.TensorOperator`.
2. `cargo test -p modelspec-runtime --test gradients_dense add_rows`: the gradients of an `mse`
   loss over `add_rows(matmul(x, w), b)` with respect to `w` and `b`, `N = 5`, `H = 3`, are within
   `TOL_F64` in `f64` and `TOL_F32` in `f32` of central differences.
3. `cargo test -p modelspec-runtime --test gradients_dense gelu`: the gradient of an `mse` loss
   over `gelu(matmul(x, w))` with respect to `w` passes the same check, with pre-activations drawn
   across `[-3, 3]` so both tails and the region near 0 are covered.
4. `cargo test -p modelspec-runtime gelu_values`: the interpreter value of `gelu` at
   `-3, -1, -0.5, 0, 0.5, 1, 3` equals the tanh form evaluated in `f64` within `1e-6` relative.
5. `cargo test -p modelspec-cli --test check_dense`: `add_rows` with a `b` of shape `[H + 1]`, and
   `add_rows` with an `x` of rank 1, exit 1 with an `E-TYPE` code at the call.
6. `task docs-check` exits 0, and `website/docs/reference/operators.md` lists `add_rows` and
   `gelu`.

## Out of scope

- The exact (erf) GELU.
- General broadcasting.
- Vulkan kernels (`story:bert-vulkan-kernels-rowwise`).
