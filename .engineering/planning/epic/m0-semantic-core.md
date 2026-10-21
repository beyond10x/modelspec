---
format: aep.planning-md/3
id: epic:m0-semantic-core
kind: epic
status: implemented
title: 'M0: semantic core (parse, check, tensor IR, autodiff, CPU training)'
relations:
- informed_by: executable-system-specification:modelspec-contracts
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-21T14:53:48Z", actor: "human:timo", revision: 4}
---
# M0: semantic core

## Outcome

A `.mspec` file in the restricted M0 subset is parsed, type- and shape-checked, lowered to typed
tensor SSA IR, differentiated in reverse mode and trained with SGD by a CPU reference interpreter.
No GPU, no performance target (DESIGN.md § 1, row M0; § 11, iteration 1).

## In scope

- M0 language subset: one model, static dimensions, `f32`, `param`, local `let`, `matmul`, `add`,
  `sub`, `mul`, `relu`, `mean`, `mse`, SGD (DESIGN.md § 2, below the first example).
- Lexer, parser, name resolution, type/shape/effect checks, Typed HIR, Tensor IR, verifier
  (DESIGN.md § 3, § 3.1–3.3).
- Reverse-mode AD with the gradient rules of DESIGN.md § 4; relu derivative at 0 is 0.
- CPU interpreter and SGD; CLI `check`, `ir`, `train --backend cpu` (DESIGN.md § 8).
- Workspace of three crates; `modelspec-core` builds and tests without Vulkan (DESIGN.md § 7).

## Acceptance

- `modelspec check` passes on `examples/linear_regression.mspec` and rejects a `matmul` contracting
  mismatch with `E-TYPE-012` (DESIGN.md § 8.1).
- Finite-difference gradient checks in `f64` pass for every M0 op and for a full linear graph
  (DESIGN.md § 4, § 9 gate 3).
- Seeded linear-regression training from `examples/linear_regression.mspec` decreases the loss and
  approaches the known solution (DESIGN.md § 9 gate 4).

## Specification

The checkpoint and training-run contracts are typed in `spec/domains/checkpoint.yaml` and
`spec/domains/training.yaml` (artifact `executable-system-specification:modelspec-contracts`).
M0 does not require checkpoints (they arrive in M1).
