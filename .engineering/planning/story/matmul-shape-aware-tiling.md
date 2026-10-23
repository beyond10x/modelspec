---
format: aep.planning-md/3
id: story:matmul-shape-aware-tiling
kind: story
status: draft
title: Matmul tiling chosen by shape, deterministic
relations:
- depends_on: story:profile-begin-stamp-after-barrier
scope:
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
revision: 2
---
# Matmul tiling chosen by shape, deterministic

## Why

Review of 0.7.1 (finding 3, its deterministic part) and Review of 0.5.0 (finding P7), at main
496e98e. The matmul tile fixes a workgroup of 128 and clamps the output columns per invocation to 16
whatever the shape (`crates/modelspec-core/src/ir/kernel.rs:195-210`). The workgroup strides in
shared memory are unpadded and unmeasured (`crates/modelspec-runtime/src/spirv/emit.rs:1391`,
`:1555-1573`, `:1567-1571`), so large matmuls may hit bank conflicts.

Outcome: the matmul tile, including shared-memory padding, depends on `m`, `k` and `n`, and large
shapes run closer to the device's throughput, without changing a bit.

## Acceptance

- **sweep-median**: the `mlp_sweep` median step time at b8192 h4096 is reported against 31067.1 µs
  (0.6.0, b2f9553) and falls.
- **per-matmul-gflops**: per-matmul GFLOP/s from `--profile` is reported before and after for every
  matmul site of that point and of the MNIST example.
- **padded-strides-measured**: padded shared-memory strides are kept only with a measured `--profile`
  delta on the MNIST matmul; otherwise the story records the figure and keeps the unpadded stride.
- **same-bits**: matmul outputs stay bit-identical to the CPU oracle; one invocation accumulates in
  ascending `k`, and the contracting index is not split (no split-K, DESIGN.md:281).
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`

## Out of scope

- FMA contraction and cooperative matrices, which story:fast-math-mode holds.
- The structural k11 change in story:weight-grad-matmul-structural.
