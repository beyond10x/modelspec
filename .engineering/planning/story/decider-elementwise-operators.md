---
format: aep.planning-md/3
id: story:decider-elementwise-operators
kind: story
status: draft
title: 'Elementwise operators: sqrt, abs and negative float literals'
summary: sqrt and abs as source operators and a negative float literal as an operand, gradient-checked, with Vulkan kernels held to the CPU reference.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
- supersedes: story:decider-small-operators
- depends_on: story:decider-loss-operators
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/check_elementwise_ops.rs
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
  path: crates/modelspec-core/src/syntax/parser.rs
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
  path: crates/modelspec-runtime/tests/gradients_elementwise_ops.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_elementwise_ops.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 3
---
# Elementwise operators: sqrt, abs and negative float literals

`depends_on: story:decider-loss-operators`; lands fourth of the epic's eight, see the epic's
§ Landing order. Split out of `story:decider-small-operators` (S2), which it supersedes with
`story:decider-shape-operators` and `story:decider-loss-operators`.

## Outcome

Three additions to the source language, each on the CPU interpreter, each with a Vulkan kernel for
its forward and for its gradient held to the CPU reference where it computes anything, and each
differentiable in its `f32` operand:

1. `sqrt(x)`: elementwise, gradient `0.5 / sqrt(x)`.
2. `abs(x)`: elementwise, gradient `sign(x)`, `0` at `0`.
3. A negative float literal as an operand (`mul(d, -1.0)`), not only as a named argument; a unary
   minus on anything but a float literal stays E-SYNTAX-009. No new code.

## Context

- Gap rows 5–10, shared with `story:decider-shape-operators` and `story:decider-loss-operators`,
  and the workaround `decider.mspec` carries here: `sub(0.0, x)` for a sign.
- Decider design § 5.2 (the head over `[s_t ; ŝ ; ‖…‖]`).
- Source operators: `OPERATORS` in `crates/modelspec-core/src/resolve.rs`; shape and gradient
  rules in `crates/modelspec-core/src/ops.rs`; `TensorOp` in `ir/tensor.rs` already has an
  internal `Sqrt` (the AdamW subgraph, `ir/tensor.rs:240`). A negative literal is parsed only after
  `name =` today (`crates/modelspec-core/src/syntax/parser.rs:894-899`).

## Spec first

`spec/domains/observatory.yaml` `TensorOperator` gains the variants the implementation adds
(`Abs`, a source-level `Sqrt` where the internal one is not reused, and their internal gradient
operators) before any code; then `task spec-validate`, `task schema-generate` (both `schema/`
trees) and `task ui-types` (`ui/src/generated/`).

## Scope

- `crates/modelspec-core/src/`: `resolve.rs`, `ops.rs`, `types.rs`, `ir/tensor.rs`,
  `ir/print.rs`, `ir/kernel.rs`, `schedule.rs`, `artifact.rs`, `autodiff.rs`,
  `syntax/parser.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`, `crates/modelspec-runtime/src/spirv/emit.rs`.
- `spec/domains/observatory.yaml`, `crates/modelspec-observatory/schema/`,
  `crates/modelspec-runtime/schema/`, `ui/src/generated/`.
- Tests (new): `crates/modelspec-runtime/tests/gradients_elementwise_ops.rs`,
  `crates/modelspec-runtime/tests/vulkan_elementwise_ops.rs`,
  `crates/modelspec-cli/tests/check_elementwise_ops.rs` with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `DESIGN.md` § 2.2, § 4, § 8.1; `crates/modelspec-docs/src/operators.rs`, then `task
  docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

- **gradients-elementwise-ops**: `cargo test -p modelspec-runtime --test gradients_elementwise_ops`
  checks the gradient of `sqrt` (on positive inputs) and `abs` (away from 0) against central
  differences, within `TOL_F64` in `f64` and `TOL_F32` in `f32`, and that the gradient of `abs` at
  0 is exactly 0.
- **vulkan-elementwise-ops**: `cargo test -p modelspec-runtime --features vulkan --test
  vulkan_elementwise_ops -- --ignored` runs each kernel on the device against the CPU reference:
  the forward of `abs` and of `mul(d, -1.0)` bit-identical, the forward of `sqrt` within
  `TOL_F32`, every gradient within `TOL_F32`; `spirv_val` (in `task test`) validates every new
  module.
- **check-elementwise-ops**: `cargo test -p modelspec-cli --test check_elementwise_ops`:
  `mul(d, -1.0)` checks and evaluates on the CPU bit-identical to `sub(0.0, d)`; `mul(-x, y)` with
  `x` a name stays E-SYNTAX-009.
- **elementwise-ops-equal-workarounds**: in the same test, a fixture computing the sign flip both
  ways (`mul(d, -1.0)` and the 0.21.0 `sub(0.0, d)`) gives a bit-identical loss on the CPU.
- `task ci` exits 0, including `task docs-check`, `task spec-validate`, `task schema-check` and
  `task ui-types-check`.

## Out of scope

- `min`, `max`, `where`, a general unary minus on expressions; a negative integer literal.
- The shape operators (`story:decider-shape-operators`) and the loss operators
  (`story:decider-loss-operators`).
- `stop_gradient`, `sin`, `cos`, SIGReg: `epic:jepa-operators`.
- Rewriting the decider fixture: the epic's closing check.
