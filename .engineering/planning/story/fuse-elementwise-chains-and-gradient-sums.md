---
format: aep.planning-md/3
id: story:fuse-elementwise-chains-and-gradient-sums
kind: story
status: draft
title: Fuse elementwise chains and gradient sums
relations:
- informed_by: story:m1-fuse-elementwise
scope:
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit.rs
revision: 3
---
# Fuse elementwise chains and gradient sums

## Why

Review of 0.7.1 (findings 2 and 9), at main 496e98e. The scheduler fuses only the cases its header
lists (`crates/modelspec-core/src/schedule.rs:4-23`); story:m1-fuse-elementwise (implemented) left
backward `ReluGrad` fusion out. Autodiff sums a value's adjoint contributions as a chain of binary
`Add`s (`crates/modelspec-core/src/autodiff.rs:511-516`, used at `:135`), each its own dispatch and
barrier on Vulkan.

Outcome: chains of elementwise ops, including `ReluGrad` and an adjoint's `Add` chain, run as one
dispatch each.

## Acceptance

- **dispatches-per-step**: dispatch count per step on the MNIST example and on `mlp_sweep` at b8192
  h4096 is reported before and after, and falls.
- **barriers-per-step**: barrier count per step is reported the same way, and falls.
- **same-bits**: the per-step loss and every parameter after 1 800 steps are bit-identical to before;
  fused kernels keep `NoContraction` and the original addition order (DESIGN.md:307).
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/autodiff.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`

## Out of scope

- Reassociating any sum; the order stays left to right.
- Fusion into matmul beyond the existing `relu` epilogue.
