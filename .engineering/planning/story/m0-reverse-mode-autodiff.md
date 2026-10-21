---
format: aep.planning-md/3
id: story:m0-reverse-mode-autodiff
kind: story
status: active
title: Reverse-mode autodiff with f64 finite-difference gradient checks
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
- depends_on: story:m0-tensor-ir-core
- depends_on: story:m0-cpu-interpreter
scope:
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/training.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/ops.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients.rs
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 8, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"review_outcome":1}}}
---
# Reverse-mode autodiff with f64 finite-difference gradient checks

## Context

Generates one backward graph on tensor SSA when a training program is compiled, not a runtime tape
(DESIGN.md § 4). Implements DESIGN.md § 4 (the gradient-rule table, relu derivative at 0 defined as
0, targets receive no gradient, each incoming adjoint accumulated exactly once, gradients in `f32`),
§ 3.1 Training IR (`Forward`, `Loss`, `Gradient` subgraphs as analysis tags, not execution
boundaries), § 3.3 pass 4, and DESIGN.md § 9 gate 3. Owns the second epic acceptance bullet.
Works on builder-made programs, so it does not wait for the front end.

## Lands in

- `crates/modelspec-core/src/ops.rs`: one gradient rule per op, as the rows of the DESIGN.md § 4
  table (`Add`, `Sub`, `Mul` including its rank-0 operand forms, `MatMul`, `Relu`, `ReduceMean`,
  `Mse`; `ReduceSum` and `Transpose` as their linear adjoints).
- `crates/modelspec-core/src/autodiff.rs`: reverse-mode pass from a scalar loss over loss-reachable
  trainable parameters only; adjoints of shared values summed once per use; emits a gradient output
  per `ParamId`, none for inputs or targets.
- `crates/modelspec-core/src/ir/training.rs`: `TrainingProgram` holding a field
  `program: TensorProgram` (the SSA graph, whose outputs are the loss followed by one gradient per
  parameter in `ParamId` order), the subgraph tag of each node, the loss value and the gradient value
  per parameter. `TrainingProgram` and `TensorProgram` share one SSA container: the interpreter keeps
  its single entry point over `&TensorProgram`, and a caller runs a training program by passing
  `&training.program`. No interpreter overload or accessor is added.
- `crates/modelspec-core/src/ir/verify.rs`: Training IR checks (every gradient has its parameter
  type; every `Gradient` node depends on the loss).
- `crates/modelspec-runtime/tests/gradients.rs`: central finite differences evaluated in `f64` on
  the CPU interpreter, inputs drawn from a fixed test-local seed and kept at least 1e-2 away from relu
  kinks. Each gradient program is evaluated in `f64` (relative error at most 1e-6) and in `f32`
  (relative error at most 1e-3) against the `f64` differences.

Graphs checked: each M0 op alone (`add`, `sub`, `mul` with same-shape and rank-0 operands, a
non-square `matmul`, `relu`, `mean`, `mse`); the full linear-regression graph
`mse(matmul(x, w), y)` with the shapes of `examples/linear_regression.mspec`; a two-layer relu graph
`mse(matmul(relu(matmul(x, w1)), w2), y)` written without modules; and shared-use graphs
(`mul(a, a)`, `add(h, h)`) for adjoint accumulation (DESIGN.md § 9 gate 3). A unit test asserts the
relu gradient at exactly 0 is 0.

## Out of scope

- The SGD optimizer subgraph, effect tokens, parameter versioning and Training IR printing (story
  `m0-cpu-sgd-training`).
- Lowering from source and the `ir` verb (stories `m0-lower-tensor-ir`, `m0-cpu-sgd-training`).
- Recomputation or checkpoint planning (DESIGN.md § 4, after correctness); user-defined `vjp`
  (DESIGN.md § 12).

## Domain relations

None. Gradient rules and Training IR are language and compiler constructs, not ESS-typed contracts
(AGENTS.md).

## Acceptance

`cargo test -p modelspec-runtime --test gradients` passes, covering each M0 op, the full linear
graph, the two-layer relu graph and the shared-use graphs within the tolerances above.
