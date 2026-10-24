---
format: aep.planning-md/3
id: story:vulkan-buffer-liveness-suballocation
kind: story
status: draft
title: Vulkan buffers reused by liveness and suballocated from few allocations
relations:
- depends_on: story:fuse-elementwise-chains-and-gradient-sums
- informed_by: story:m1-kernel-plan
- depends_on: story:vulkan-step-counters
scope:
- confidence: cited
  path: crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir
- confidence: inferred
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/memory.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: website/docs/guides/inspect-a-run.md
revision: 5
---
# Vulkan buffers reused by liveness and suballocated from few allocations

## Why

Review of 0.7.1 (finding 1), at main 496e98e. The scheduler gives every stored value its own buffer
("There is no buffer reuse", `crates/modelspec-core/src/schedule.rs:25`); the runner creates them one
by one (`crates/modelspec-runtime/src/vulkan/runner.rs:489-494`), and each gets its own
`vkAllocateMemory` (`crates/modelspec-runtime/src/vulkan/memory.rs:190`). story:m1-kernel-plan
deferred liveness-based reuse to M2 (its line 208). At large batch and width the peak device memory
is the sum of every intermediate.

Sharing storage changes the `buffer bN` list and the view bindings that the kernel golden
(`crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir:1`) and the guide's kernel lines
(`website/docs/guides/inspect-a-run.md`) pin byte for byte.

The two counts it reads are the `device allocations` and `peak device bytes` counters of
story:vulkan-step-counters, which this story follows.

Outcome: values whose lifetimes do not overlap share storage, and buffers are suballocated from at
most `MAX_DEVICE_ALLOCATIONS` device allocations per run.

## Acceptance

- **peak-device-bytes**: on `mlp_sweep` at b8192 h4096, `peak device bytes` from `--stats`
  (story:vulkan-step-counters) is reported before and after, and falls.
- **allocate-count**: `crates/modelspec-runtime/src/vulkan/memory.rs` defines
  `pub const MAX_DEVICE_ALLOCATIONS: u64 = 8`, and `device allocations` from `--stats` is at most
  that constant on the MNIST example and on the sweep point; it is reported before and after.
- **same-loss-bits**: the per-step loss is bit-identical to before on the MNIST example and the
  sweep point.
- **verifier-checks-aliasing**: `verify_plan` refuses a plan where two values sharing storage are
  live at once.
- **goldens-regenerated**: the kernel golden and the guide's kernel lines are regenerated in the
  same change, and `crates/modelspec-cli/tests/ir_kernel.rs` and
  `crates/modelspec-cli/tests/adversary_guide_kernel_lines.rs` pass.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/verify.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-runtime/src/vulkan/memory.rs`
- `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`
- `website/docs/guides/inspect-a-run.md`

## Out of scope

- Reuse on the CPU interpreter, which story:cpu-prepared-training-step holds.
- Changing which values are fused away; that is story:fuse-elementwise-chains-and-gradient-sums,
  which this story follows so liveness is computed on the fused schedule.
- Adding the counters, which story:vulkan-step-counters holds.
