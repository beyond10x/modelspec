---
format: aep.planning-md/3
id: story:m0-cpu-interpreter
kind: story
status: implemented
title: CPU reference interpreter over Tensor IR
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
- depends_on: story:m0-tensor-ir-core
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/tensor.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/cpu_adversary.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/cpu_golden.rs
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:26Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:26Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "active", to: "implemented", at: "2026-10-21T14:53:41Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"test_result":1,"review_outcome":2,"verification":1}}}
---
# CPU reference interpreter over Tensor IR

## Context

The CPU oracle that every later backend is compared against (DESIGN.md § 0 invariant 1; ADR-004;
AGENTS.md "Correctness before speed"). It executes a verified `TensorProgram` and nothing else; it
never reparses source (DESIGN.md § 7.1). Implements DESIGN.md § 6 (the CPU backend implements the
same observable semantics), § 2.2 forward semantics and the "explicit specified reduction order
policy", and DESIGN.md § 9 gate 2 (CPU golden values for each tensor op).

## Lands in

Crate `modelspec-runtime`, files created empty by the foundation story:

- `src/cpu/tensor.rs`: dense row-major host tensor, generic over the element type so the same code
  runs in `f32` and in `f64` (the gradient story evaluates finite differences in `f64`, DESIGN.md § 4).
- `src/cpu/interpreter.rs`: evaluates a `TensorProgram` from named inputs and a parameter-value map;
  runs the core verifier first; checks every supplied tensor's shape against its declared type (the
  element type is the interpreter's type parameter, so a program typed `DType::F32` also runs in
  `f64` for the gradient checks); returns the
  program outputs and, on request, every intermediate value. Implements every op of the registry,
  including the internal gradient ops `Transpose`, `ReluGrad` and `BroadcastScalar`. Errors are a
  `thiserror` type (DESIGN.md § 8.1 "runtime" class).
- Reduction order policy: sums accumulate sequentially in row-major index order in the element
  type; `matmul` accumulates over the contracting index in ascending order; `mean` is that sum divided
  by the element count; `mse` follows DESIGN.md § 2.2. Record this policy as one sentence in
  DESIGN.md § 2.2 and bump the DESIGN.md revision line (AGENTS.md rule on semantics changes).
- Tests: `crates/modelspec-runtime/tests/cpu_golden.rs`, building programs with the core builder.

## Out of scope

- Lowering from source; gradients and their checks (story `m0-reverse-mode-autodiff`); SGD, the
  training loop, parameter initialization and data (story `m0-cpu-sgd-training`).
- Training IR effect tokens and state versions: they are metadata in `ir/training.rs`, never
  interpreter ops, and the host runner applies parameter writes (story `m0-cpu-sgd-training`). No
  later M0 story edits `src/cpu/interpreter.rs`.
- Performance, threading, SIMD.

## Domain relations

None. The interpreter reads Tensor IR, a language construct, not an ESS-typed contract (AGENTS.md).

## Acceptance

`cargo test -p modelspec-runtime --test cpu_golden` passes, comparing every registry op (including
a non-square `[2, 3] x [3, 4]` matmul, `transpose`, `sum`, `mean` and `mse`) against hand-computed
values chosen to be exactly representable, so `f32` and `f64` results equal them bit for bit.
