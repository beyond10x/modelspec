---
format: aep.planning-md/3
id: story:jepa-stop-gradient
kind: story
status: active
title: 'stop_gradient: a value passes forward and no gradient flows back'
summary: A source operator whose backward rule is zero, removed after autodiff, with E-TYPE-026 for a parameter that reaches the loss only through it.
relations:
- decomposes: epic:jepa-operators
- serves: vision:modelspec
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests
- confidence: cited
  path: crates/modelspec-core/src
- confidence: cited
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients_stop_gradient.rs
- confidence: cited
  path: schema
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 12
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T10:45:52Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-28T10:45:52Z", actor: "human:timo", revision: 3}
---

`id: story:jepa-stop-gradient` · depends_on: none · size: about half a day

## Outcome

`stop_gradient(x)` returns `x` unchanged in the forward pass and passes no adjoint back to `x`.
The context branch of a JEPA can therefore read the online encoder's output as a fixed target in the
same training step. It costs no kernel: a pass after autodiff replaces each use by its operand
before the CPU interpreter or the scheduler sees the program.

## Context

- The source operators are the table `OPERATORS` at `crates/modelspec-core/src/resolve.rs:64`. The
  test `every_source_operator_is_a_registry_op` (`types.rs:1690-1715`) pins that list.
- Shape rule: the identity group at `crates/modelspec-core/src/ops.rs:353`. Gradient rules:
  `ops::gradient` (`ops.rs:620`, its table at `:585-619`). Autodiff only walks nodes that depend
  on a parameter read (`autodiff.rs:1-13`).
- The verifier refuses a parameter the loss reads but that has no gradient (`ir/verify.rs:322`).
  `TrainingProgram::gradients` holds exactly the parameters the loss reads (`ir/training.rs:76-80`).
  So a parameter that reaches the loss only through `stop_gradient` must be refused by the checker,
  not left to the verifier. The checker already tracks parameter reads through dataflow for
  E-TYPE-021 (`types.rs:28-33`, code at `:175`). E-TYPE-025 is the last code in use.
- Passes run in `compile.rs:72` (`dce::eliminate`, `passes/dce.rs:16`).

## Scope

- `crates/modelspec-core/src/resolve.rs`, `ops.rs`, `ir/tensor.rs` (`StopGradient`), `ir/print.rs`,
  `ir/verify.rs`, `types.rs` (E-TYPE-026: a parameter that reaches the loss only through
  `stop_gradient`, reported at the call), `passes/mod.rs`, `passes/stop_gradient.rs` (new),
  `compile.rs`.
- `crates/modelspec-runtime/tests/gradients_stop_gradient.rs` (new).
- `crates/modelspec-cli/tests/check_stop_gradient.rs` (new) with its fixtures under
  `crates/modelspec-cli/tests/fixtures/`. Also
  `tests/language/check/E-TYPE-026-parameter-only-through-stop-gradient.mspec`.
- `spec/domains/observatory.yaml:203-204` (`StopGradient`), then `task schema-generate` (both
  `schema/` trees) and `task ui-types` (`ui/src/generated/`).
- DESIGN.md: § 2.2 (shape rule), § 4 (gradient table row), § 8.1 (E-TYPE-026) and the revision
  line. `crates/modelspec-docs/src/operators.rs`, then `task docs-generate`
  (`website/docs/reference/operators.md`, `diagnostics.md`, `website/data/`). One line in
  `CHANGELOG.md`.

## Acceptance

`cargo test -p modelspec-runtime --test gradients_stop_gradient` passes on this loss:
`mse(add(matmul(x, w), stop_gradient(matmul(x, w))), y)`. The gradient of `w` must equal, within
`TOL_F64` in `f64` and `TOL_F32` in `f32`, the central differences of the same function with the
stopped term held as a constant. That gradient is half of what the loss without `stop_gradient`
gives.

## Done when

- `modelspec check` on a file whose only read of a parameter is under `stop_gradient` exits 1 with
  E-TYPE-026 at the call (`check_stop_gradient`).
- `modelspec ir --stage tensor` prints `StopGradient`, and `modelspec ir --stage training` has no
  such node.
- `task ci` is green, including `task docs-check`, `task spec-validate` and `task schema-check`.

## Out of scope

- Frozen (non-trainable) parameters, and `ema_update` with its EMA copies.
- Any Vulkan kernel: the pass removes the op before scheduling.

---

