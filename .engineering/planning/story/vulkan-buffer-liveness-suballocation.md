---
format: aep.planning-md/3
id: story:vulkan-buffer-liveness-suballocation
kind: story
status: draft
title: Vulkan buffers reused by liveness and suballocated from few allocations
relations:
- depends_on: story:fuse-elementwise-chains-and-gradient-sums
- informed_by: story:m1-kernel-plan
scope:
- confidence: inferred
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/memory.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
revision: 3
---
# Vulkan buffers reused by liveness and suballocated from few allocations

## Why

Review of 0.7.1 (finding 1), at main 496e98e. The scheduler gives every stored value its own buffer
("There is no buffer reuse", `crates/modelspec-core/src/schedule.rs:25`); the runner creates them one
by one (`crates/modelspec-runtime/src/vulkan/runner.rs:489-494`), and each gets its own
`vkAllocateMemory` (`crates/modelspec-runtime/src/vulkan/memory.rs:190`). story:m1-kernel-plan
deferred liveness-based reuse to M2 (its line 208). At large batch and width the peak device memory
is the sum of every intermediate.

Outcome: values whose lifetimes do not overlap share storage, and buffers are suballocated from a
small number of device allocations.

## Acceptance

- **peak-device-bytes**: on `mlp_sweep` at b8192 h4096, the peak device bytes per step are reported
  before and after, and fall.
- **allocate-count**: the number of `vkAllocateMemory` calls per run is reported before and after,
  and is bounded by a constant independent of the number of values.
- **same-loss-bits**: the per-step loss is bit-identical to before on the MNIST example and the
  sweep point.
- **verifier-checks-aliasing**: `verify_plan` refuses a plan where two values sharing storage are
  live at once.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/verify.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-runtime/src/vulkan/memory.rs`

## Out of scope

- Reuse on the CPU interpreter, which story:cpu-prepared-training-step holds.
- Changing which values are fused away; that is story:fuse-elementwise-chains-and-gradient-sums,
  which this story follows so liveness is computed on the fused schedule.
