---
format: aep.planning-md/3
id: story:mean-mse-adjoints-divide-by-n
kind: story
status: draft
title: Mean and mse adjoints divide by n
scope:
- confidence: inferred
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ops.rs
revision: 3
---
# Mean and mse adjoints divide by n

## Why

Review of 0.5.0 (finding C4), still open at main 496e98e. The adjoints of `mean` and `mse` multiply
by `Scalar(1/n)` and `Scalar(2/n)` (`crates/modelspec-core/src/ops.rs:261-285`). For an `n` that is
not a power of two, `1/n` is rounded before the product, so the gradient differs in its last bits
from the `G / n` and `(Y - T) * G * 2 / n` that DESIGN.md § 4 states.

Outcome: the gradient bits of `mean` and `mse` equal the formula DESIGN.md § 4 gives.

## Acceptance

- **mean-adjoint-bits**: for `n` = 3, 10 and 100, the `mean` gradient's bits equal `G / n`
  computed in `f32`.
- **mse-adjoint-bits**: for the same `n`, the `mse` gradient's bits equal `(Y - T) * G * 2 / n`
  in that order.
- **vulkan-follows**: the Vulkan backend's gradients for those programs match the CPU's in the
  class DESIGN.md § 5.4 assigns them.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ops.rs`
- `crates/modelspec-core/src/autodiff.rs`

## Out of scope

- Changing the order DESIGN.md § 4 states; if the formula itself should change, DESIGN.md changes
  first.
