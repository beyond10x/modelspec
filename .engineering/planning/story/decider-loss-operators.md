---
format: aep.planning-md/3
id: story:decider-loss-operators
kind: story
status: draft
title: 'Loss operators: smooth_l1 and masked cross_entropy'
summary: smooth_l1 with a beta threshold and an optional class mask on cross_entropy, gradient-checked, with Vulkan kernels held to the CPU reference.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
- supersedes: story:decider-small-operators
- depends_on: story:decider-shape-operators
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/check_loss_ops.rs
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
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/gradients_loss_ops.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_loss_ops.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 3
---
# Loss operators: smooth_l1 and masked cross_entropy

`depends_on: story:decider-shape-operators`; lands third of the epic's eight, see the epic's
§ Landing order. Split out of `story:decider-small-operators` (S2), which it supersedes with
`story:decider-shape-operators` and `story:decider-elementwise-operators`.

## Outcome

Two additions to the source language, each on the CPU interpreter, each with a Vulkan kernel for
its forward and for its gradient held to the CPU reference, and each differentiable in its `f32`
operands:

1. `smooth_l1(a, b, beta = F)`: the mean over all elements of the Huber-style smooth-L1 of
   `a − b` with threshold `beta > 0`, a rank-0 result like `mse`. A `beta` that is not greater
   than 0 is refused at the call with E-TYPE-030.
2. `cross_entropy(logits, labels, weights, mask)`: an optional fourth operand, `f32` of the
   logits' shape, `1` for a class that takes part and `0` for one that does not; a masked class
   gets probability 0 and gradient 0. A mask whose shape is not the logits' is refused at the call
   with E-TYPE-031. A weighted row whose label is a masked class gives a non-finite loss, which
   the existing non-finite refusal reports.

## Context

- Gap rows 5–10, shared with `story:decider-shape-operators` and
  `story:decider-elementwise-operators`, and the workarounds `decider.mspec` carries for these
  two: smooth-L1 from `relu`, `sub` and `mul`, and an additive `-1e9` bias for absent candidates.
- Decider design § 5.3 step 1 (smooth-L1 in embedding space), step 2 (cross-entropy over
  candidates; not every template applies to every decision, § 4.3).
- Source operators: `OPERATORS` in `crates/modelspec-core/src/resolve.rs`; shape and gradient
  rules in `crates/modelspec-core/src/ops.rs`; the `cross_entropy` checks in
  `crates/modelspec-core/src/types.rs` (E-TYPE-003, E-TYPE-004, E-TYPE-018, E-TYPE-021 today).
- Codes: `story:decider-shape-operators` takes E-TYPE-027 to E-TYPE-029, so this story takes
  E-TYPE-030 and E-TYPE-031.

## Spec first

`spec/domains/observatory.yaml` `TensorOperator` gains the variants the implementation adds
(`SmoothL1`, the masked cross-entropy and their internal gradient operators) before any code; then
`task spec-validate`, `task schema-generate` (both `schema/` trees) and `task ui-types`
(`ui/src/generated/`).

## Scope

- `crates/modelspec-core/src/`: `resolve.rs`, `ops.rs`, `types.rs`, `ir/tensor.rs`,
  `ir/print.rs`, `ir/kernel.rs`, `schedule.rs`, `artifact.rs`, `autodiff.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`, `crates/modelspec-runtime/src/spirv/emit.rs`,
  `crates/modelspec-runtime/src/spirv/emit/rowwise.rs`.
- `spec/domains/observatory.yaml`, `crates/modelspec-observatory/schema/`,
  `crates/modelspec-runtime/schema/`, `ui/src/generated/`.
- Tests (new): `crates/modelspec-runtime/tests/gradients_loss_ops.rs`,
  `crates/modelspec-runtime/tests/vulkan_loss_ops.rs`,
  `crates/modelspec-cli/tests/check_loss_ops.rs` with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `DESIGN.md` § 2.2, § 4, § 8.1; `crates/modelspec-docs/src/operators.rs`, then `task
  docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

- **gradients-loss-ops**: `cargo test -p modelspec-runtime --test gradients_loss_ops` checks the
  gradient of every `f32` operand of `smooth_l1` (on both sides of `beta`) and of masked
  `cross_entropy` against central differences, within `TOL_F64` in `f64` and `TOL_F32` in `f32`;
  the masked classes' gradient is exactly 0.
- **vulkan-loss-ops**: `cargo test -p modelspec-runtime --features vulkan --test vulkan_loss_ops
  -- --ignored` runs each kernel on the device against the CPU reference: the forward of
  `smooth_l1` and of masked `cross_entropy` within `TOL_F32`, every gradient within `TOL_F32`, the
  masked classes' gradient exactly 0; `spirv_val` (in `task test`) validates every new module.
- **check-loss-ops**: `cargo test -p modelspec-cli --test check_loss_ops`: `beta = 0.0` is refused
  at the call with E-TYPE-030, and a mask whose shape is not the logits' with E-TYPE-031.
- **loss-ops-equal-workarounds**: in the same test, a fixture computing smooth-L1 and masked
  cross-entropy both ways (the operator, and the 0.21.0 workaround the decider file uses) gives
  the same loss on the CPU within `TOL_F32`.
- `task ci` exits 0, including `task docs-check`, `task spec-validate`, `task schema-check` and
  `task ui-types-check`.

## Out of scope

- `smooth_l1` with a per-element `beta` or a reduction other than the mean.
- The shape operators (`story:decider-shape-operators`) and the elementwise operators
  (`story:decider-elementwise-operators`).
- `stop_gradient`, `sin`, `cos`, SIGReg: `epic:jepa-operators`.
- Rewriting the decider fixture: the epic's closing check.
