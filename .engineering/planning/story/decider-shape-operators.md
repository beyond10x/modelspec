---
format: aep.planning-md/3
id: story:decider-shape-operators
kind: story
status: draft
title: 'Shape operators: reshape and last-axis concat'
summary: reshape and two-operand last-axis concat, gradient-checked, with Vulkan kernels held to the CPU reference.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
- supersedes: story:decider-small-operators
- depends_on: story:decider-dense-named-data
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/check_shape_ops.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ops.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-observatory/schema
- confidence: cited
  path: crates/modelspec-runtime/schema
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/gradients_shape_ops.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_shape_ops.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 3
---
# Shape operators: reshape and last-axis concat

`depends_on: story:decider-dense-named-data` (S1); lands second of the epic's eight, see the epic's
§ Landing order. Split out of `story:decider-small-operators` (S2), which it supersedes with
`story:decider-loss-operators` and `story:decider-elementwise-operators`.

## Outcome

Two additions to the source language, each on the CPU interpreter, each with a Vulkan kernel for
its forward and for its gradient held to the CPU reference, and each differentiable in its `f32`
operands:

1. `reshape(x, shape = [d1, …])`: row-major, the element count unchanged; its gradient is the
   reverse reshape. A shape whose element count differs from `x`'s is refused at the call with
   E-TYPE-027.
2. `concat(a, b)`: on the last axis of two rank-2 operands with equal leading dimension; its
   gradient is the split. An operand that is not rank 2 is refused at the call with E-TYPE-028,
   two operands of unequal leading dimension with E-TYPE-029.

## Context

- Gap rows 5–10, shared with `story:decider-loss-operators` and
  `story:decider-elementwise-operators`, and the workarounds `decider.mspec` carries for these
  two: two selection matmuls and a broadcast matmul for the reshape (`fold_rows`, `fold_cols`, `fold_bcast`; ≈ 138 MFLOP a
  step and 4.3 MB of data for `fold_rows` alone), and a split first-layer weight for the
  concatenation.
- Decider design § 5.2 (predictor input `s_t` concatenated with a candidate action embedding; the
  head over `[s_t ; ŝ ; ‖…‖]`).
- Source operators: `OPERATORS` in `crates/modelspec-core/src/resolve.rs`; shape and gradient
  rules in `crates/modelspec-core/src/ops.rs`.
- Codes: E-TYPE-025 is the last on main; E-TYPE-026 is taken by `story:jepa-stop-gradient`
  (branch `plan/jepa-operators`), so this story takes E-TYPE-027 to E-TYPE-029.
- The planned `story:jepa-stop-gradient` touches the same files (`resolve.rs`, `ops.rs`,
  `ir/tensor.rs`, `spec/domains/observatory.yaml`); it is not in this store's main yet, so
  `waves` cannot see the collision. Land them one after the other.

## Spec first

`spec/domains/observatory.yaml` `TensorOperator` gains the variants the implementation adds
(`Reshape`, `Concat` and their internal gradient operators) before any code; then `task
spec-validate`, `task schema-generate` (both `schema/` trees) and `task ui-types`
(`ui/src/generated/`).

## Scope

- `crates/modelspec-core/src/`: `resolve.rs`, `ops.rs`, `types.rs`, `ir/tensor.rs`,
  `ir/print.rs`, `ir/kernel.rs`, `schedule.rs`, `artifact.rs`, `autodiff.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`, `crates/modelspec-runtime/src/spirv/emit.rs`.
- `spec/domains/observatory.yaml`, `crates/modelspec-observatory/schema/`,
  `crates/modelspec-runtime/schema/`, `ui/src/generated/`.
- Tests (new): `crates/modelspec-runtime/tests/gradients_shape_ops.rs`,
  `crates/modelspec-runtime/tests/vulkan_shape_ops.rs`,
  `crates/modelspec-cli/tests/check_shape_ops.rs` with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `DESIGN.md` § 2.2, § 4, § 8.1; `crates/modelspec-docs/src/operators.rs`, then `task
  docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

- **gradients-shape-ops**: `cargo test -p modelspec-runtime --test gradients_shape_ops` checks the
  gradient of every `f32` operand of `reshape` and `concat` against central differences, within
  `TOL_F64` in `f64` and `TOL_F32` in `f32`.
- **vulkan-shape-ops**: `cargo test -p modelspec-runtime --features vulkan --test vulkan_shape_ops
  -- --ignored` runs each kernel on the device against the CPU reference: the forward of `reshape`
  and of `concat` bit-identical, every gradient within `TOL_F32`; `spirv_val` (in `task test`)
  validates every new module.
- **check-shape-ops**: `cargo test -p modelspec-cli --test check_shape_ops`: a `reshape` whose
  element count differs is refused at the call with E-TYPE-027, a `concat` of rank-1 operands with
  E-TYPE-028 and a `concat` of unequal leading dimensions with E-TYPE-029.
- **shape-ops-equal-workarounds**: in the same test, a fixture computing the reshape and the
  concatenation both ways (the operator, and the 0.21.0 workaround the decider file uses) gives
  the same loss on the CPU: the reshape and its selection matmuls bit-identical; the concatenation
  and its split first-layer weight within `TOL_F32`, because the split weight sums two matmuls in
  another order.
- `task ci` exits 0, including `task docs-check`, `task spec-validate`, `task schema-check` and
  `task ui-types-check`.

## Out of scope

- `concat` on other axes or of more than two operands; `reshape` to a symbolic shape.
- The loss operators (`story:decider-loss-operators`) and the elementwise operators
  (`story:decider-elementwise-operators`).
- `stop_gradient`, `sin`, `cos`, SIGReg: `epic:jepa-operators`.
- Rewriting the decider fixture: the epic's closing check.
