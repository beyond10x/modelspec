---
format: aep.planning-md/3
id: story:fuse-elementwise-chains-and-gradient-sums
kind: story
status: draft
title: Fuse elementwise chains and gradient sums
relations:
- informed_by: story:m1-fuse-elementwise
- depends_on: story:vulkan-step-counters
scope:
- confidence: cited
  path: crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir
- confidence: inferred
  path: crates/modelspec-cli/tests/golden/mnist_mlp.training.ir
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: website/docs/guides/inspect-a-run.md
revision: 6
---
# Fuse elementwise chains and gradient sums

## Why

Review of 0.7.1 (findings 2 and 9), at main 496e98e. The scheduler fuses only the cases its header
lists (`crates/modelspec-core/src/schedule.rs:4-23`); story:m1-fuse-elementwise (implemented) left
backward `ReluGrad` fusion out. Autodiff sums a value's adjoint contributions as a chain of binary
`Add`s (`crates/modelspec-core/src/autodiff.rs:511-516`, used at `:135`), each its own dispatch and
barrier on Vulkan.

The dispatch list this story changes is pinned byte for byte by the kernel golden
(`crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`, checked by
`crates/modelspec-cli/tests/ir_kernel.rs:36`) and by the kernel lines the guide quotes
(`website/docs/guides/inspect-a-run.md`, checked by
`crates/modelspec-cli/tests/adversary_guide_kernel_lines.rs`). A change to the adjoint sum in
`autodiff.rs` also changes the training IR golden (`crates/modelspec-cli/tests/golden/mnist_mlp.training.ir`).

The barrier count it reads is the `barriers per step` counter of story:vulkan-step-counters, which
this story follows.

Outcome: chains of elementwise ops, including `ReluGrad` and an adjoint's `Add` chain, run as one
dispatch each.

## Acceptance

- **dispatches-per-step**: `dispatches per step` from `--stats` on the MNIST example and on
  `mlp_sweep` at b8192 h4096 is reported before and after, and falls.
- **barriers-per-step**: `barriers per step` from `--stats` (story:vulkan-step-counters) is reported
  the same way, and falls.
- **same-bits**: the per-step loss and every parameter after 1 800 steps are bit-identical to before;
  fused kernels keep `NoContraction` and the original addition order (DESIGN.md:307).
- **goldens-regenerated**: the kernel golden, the training IR golden where the training IR changes,
  and the guide's kernel lines are regenerated in the same change, and their tests pass.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/autodiff.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`
- `crates/modelspec-cli/tests/golden/mnist_mlp.training.ir`
- `website/docs/guides/inspect-a-run.md`

## Out of scope

- Reassociating any sum; the order stays left to right.
- Fusion into matmul beyond the existing `relu` epilogue.
- Adding the barrier counter, which story:vulkan-step-counters holds.
