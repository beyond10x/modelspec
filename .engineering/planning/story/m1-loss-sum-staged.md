---
format: aep.planning-md/3
id: story:m1-loss-sum-staged
kind: story
status: implemented
title: The mse loss sum reads its terms through workgroup memory, in the same order
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
- depends_on: story:m1-vulkan-profile
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_loss_sum.rs
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T11:18:04Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-22T11:18:04Z", actor: "human:timo", revision: 5}
- {from: "active", to: "implemented", at: "2026-10-22T12:57:00Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# The `mse` loss sum reads its terms through workgroup memory, in the same order

## Context

`train --backend vulkan --profile` (`story:m1-vulkan-profile`, wave m1-e) on
`examples/mnist_mlp.mspec`, 72,000 steps, RTX 3090 at P0 1951 MHz: the loss sum (kernel kind
`loss_sum`, step-plan position 6) takes 29.07 µs mean per step, the largest dispatch site; a
profiled step is 90.3 µs and an unprofiled one 68.5 µs median. Positions 7 and 8 share its barrier
group at 1.91 and 1.67 µs, so the loss sum alone sets that group's length, about 27 µs above them.

The kernel is one invocation (`workgroup_size` 1, `[1, 1, 1]`, DESIGN.md § 3.1 and § 5.2): it reads
2,000 floats from device memory one after the other. Inferred, not measured: the time is load
latency, about 29 ns per term, not the 1,000 additions.

DESIGN.md § 5.4 keeps the loss sum in the bit-for-bit class because one invocation accumulates it in
row-major order from the first term. This story keeps that order and changes only how the terms
reach the invocation that adds them.

## What it changes

- The `loss_sum` kernel runs one workgroup of 64 invocations. They copy `a` and `b` (or the terms
  `(a - b) * (a - b)`, each computed by one invocation with `NoContraction`) into workgroup memory
  in coalesced loads, a workgroup barrier follows, and invocation 0 adds the 1,000 terms in
  row-major order from the first, exactly as now, then divides where the kernel divides now.
- Workgroup memory size comes from the shape at compile time. A loss whose terms do not fit the
  device's `maxComputeSharedMemorySize` from the device profile stages them in chunks, each chunk
  added in order before the next is loaded; the order of additions does not change.
- DESIGN.md § 3.1 (the workgroup size of the loss sum) and § 5.2 (the loss-sum line) say so; § 5.4
  is unchanged.

## Scope

- `crates/modelspec-core/src/ir/kernel.rs` (`workgroup_size` of `LossSum`),
  `crates/modelspec-core/src/schedule.rs` (its workgroups).
- `crates/modelspec-runtime/src/spirv/emit.rs` (the `loss_sum` body),
  `crates/modelspec-runtime/src/vulkan/dispatch.rs` if its `LossSum` arm needs the new size.
- `crates/modelspec-runtime/tests/vulkan_reduce_matmul.rs` or a new
  `crates/modelspec-runtime/tests/vulkan_loss_sum.rs`.
- `DESIGN.md` § 3.1, § 5.2 and the revision line; `CHANGELOG.md`.
- Shares `emit.rs`, `schedule.rs`, `kernel.rs`, DESIGN.md and CHANGELOG.md with
  `story:m1-fuse-elementwise`; the wave merges this one first.

## Acceptance

The loss sum keeps its bits and stops being the longest dispatch of a step.

1. Operator run on the RTX 3090: the device tests of `vulkan_reduce_matmul.rs` and `vulkan_train.rs`
   pass. Where `negotiate` gives `bit-for-bit` the loss sum before its division equals the CPU
   oracle bit for bit on the existing shapes and on a loss of 1, 63, 64, 65 and 100,000 terms (the
   last one past one chunk of workgroup memory on the RTX 3090); elsewhere within the existing
   per-operator tolerance.
2. `spirv-val --target-env vulkan1.2` accepts the emitted module (`spirv_val.rs`), and the module
   still decorates every `OpFAdd`, `OpFSub` and `OpFMul` with `NoContraction`.
3. `train examples/mnist_mlp.mspec --backend vulkan --profile` on a warm device: the loss-sum line's
   `mean_us` is at most 6 and it is not the longest dispatch line of a step. The losses on stdout
   are byte-equal to the base commit's. (Restated from "at most 5", an estimate made before the
   change: the staged kernel measures 5.620 µs at b216b5b, and the rest is invocation 0's additions
   in row-major order, which § 5.4 keeps. Accepted by the chef orchestrator on 2026-10-22.)
4. Steady-state µs per step by the ADR-007 timing method (`--stats`, no `--profile`), before and
   after on the same device: median, range and clock reported; the median drops by at least 15 µs.
5. `task ci` exits 0.

## Out of scope

- A tree or two-stage loss reduction: it changes the order and so the § 5.4 class.
- Skipping the loss sum on steps whose loss is not reported (DESIGN.md § 6.3, M2).
