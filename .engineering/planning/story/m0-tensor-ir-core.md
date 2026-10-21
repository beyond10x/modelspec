---
format: aep.planning-md/3
id: story:m0-tensor-ir-core
kind: story
status: draft
title: Tensor IR, operator registry, verifier and printer
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
- depends_on: story:m0-workspace-foundation
revision: 2
---
# Tensor IR, operator registry, verifier and printer

## Context

The typed SSA tensor IR is the one semantic core that the CPU interpreter, autodiff and (in M1) the
GPU compiler all consume (DESIGN.md § 0 invariant 1; ADR-003). This story builds it standalone,
constructed through a builder API, so the interpreter and autodiff stories can start before the
front end lowers source into it. Implements DESIGN.md § 3.1 (Tensor IR), § 3.2 (typed IDs,
`TensorType`, `TensorOp`, `TensorNode`), the shape rules of § 2.2 for the M0 ops, and the verifier
of § 3.2 and § 3 ("the verifier runs after major transformations").

## Lands in

Crate `modelspec-core`, files created empty by the foundation story:

- `src/ir/tensor.rs`: `ValueId`, `ParamId`, `DType::F32`, `Dim`, `TensorType`, `TensorOp`,
  `TensorNode`, and a `TensorProgram` (nodes, a parameter table mapping `ParamId` to canonical path,
  type and initializer, named inputs, outputs), plus a builder that infers each result type through
  the registry. Ops: `Input`, `ParamRead`, `Scalar`, `Add`, `Sub`, `Mul` (same shape, or one rank-0
  operand), `MatMul` (rank 2), `Relu`, `Transpose`, `ReduceSum`, `ReduceMean`, `Mse`, and the
  internal gradient ops DESIGN.md § 4 names: `ReluGrad` (the `where(A > 0, G, 0)` form) and
  `BroadcastScalar { shape }`.
- `src/ops.rs`: the operator registry (DESIGN.md § 2.2, "one operator registry"): per op, arity and
  shape/dtype rule returning a structured error (for `MatMul`: lhs type, rhs type, expected and got
  contracting dimension) that the checker maps to `E-TYPE-012`. Gradient rules are added by story
  `m0-reverse-mode-autodiff`.
- `src/ir/verify.rs`: definition before use (topological order), every `ParamId` declared in the
  table, operand types consistent with the registry, outputs defined, no `Dim::Symbol` left, no zero
  dimension, element and byte counts computed with checked arithmetic (DESIGN.md § 2.2, last rule).
  Verification failures are a `thiserror` type, distinct from user diagnostics (DESIGN.md § 8.1).
- `src/ir/print.rs`: readable text form with stable IDs (DESIGN.md § 8), one line per node, for
  example `%2: Tensor<f32, [8, 1]> = MatMul %0, %1`, with parameters printed by canonical path.
- Tests: `crates/modelspec-core/tests/tensor_ir.rs`, goldens beside it.

## Out of scope

- Any `Cargo.toml` or `Cargo.lock` edit: the foundation story declares `thiserror` for
  `modelspec-core`.
- Lowering from source (story `m0-lower-tensor-ir`); Training IR, effect tokens and gradient rules
  (stories `m0-reverse-mode-autodiff`, `m0-cpu-sgd-training`).
- Executing the IR (story `m0-cpu-interpreter`).
- Canonicalization, constant folding, layouts, Kernel IR (DESIGN.md § 3.1 lower levels; M1 and
  later).
- Dtypes other than `f32`; non-scalar broadcasting (DESIGN.md § 2.2).

## Domain relations

None. IR values and ops are language constructs, not ESS-typed contracts (AGENTS.md).

## Acceptance

`cargo test -p modelspec-core --test tensor_ir` passes, including verifier rejections of a use
before definition, an undeclared `ParamId` and a `MatMul` with mismatched contracting dimension,
and a printer golden for a builder-made `mse(matmul(x, w), y)` graph.
