---
format: aep.planning-md/3
id: story:m1-f-forward-matmul-tiled
kind: story
status: active
title: Tiled matmul kernel with workgroup-memory staging, measured on the forward matmul
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
- depends_on: story:m1-fuse-elementwise
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: inferred
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests
revision: 12
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T15:28:53Z", actor: "human:timo", revision: 11}
- {from: "proposed", to: "active", at: "2026-10-22T15:28:53Z", actor: "human:timo", revision: 12}
---
# Tiled matmul kernel with workgroup-memory staging, measured on the forward matmul

## Context

`train --backend vulkan --profile` on the wave m1-e head `7a488c1`, 144,000 steps, RTX 3090, P0 in
165 samples at a mean of 1951 MHz, 3 profiled runs after a warm-up (wave m1-f profile, 2026-10-22):

| position | kernel | shape | mean µs (3 runs) |
|---|---|---|---|
| 0 | `k0` `0f20c50b`, matmul with ReLU epilogue | `[100,784] x [784,64]`, 100 workgroups | 18.372 / 19.279 / 18.545 |
| 17 | `k11` `433c82fb`, matmul (dW1) | `[784,100] x [100,64]`, 784 workgroups | 6.697 / 7.218 / 6.957 |

`k0` is the longest dispatch of the step (about 18.4 of 38.8 to 40.2 µs). The matmul kernel today
(`Module::matmul` in `crates/modelspec-runtime/src/spirv/emit.rs`) runs one invocation per output
element and reads both operands from storage buffers for every `k`, with no reuse across the
workgroup. DESIGN.md § 5.2 (M2 list) already names workgroup tiling and shared-memory staging, with
one invocation accumulating each output element in ascending `k`, tile after tile, and no split-K.

## What it changes

- The matmul kernel stages tiles of both operands in workgroup memory and accumulates each output
  element in one invocation over ascending `k`, tile after tile, starting from the `k = 0` product;
  no split-K. An invocation may compute more than one output element (register blocking); each
  element keeps its own accumulation order.
- Tile and workgroup shape are chosen per kernel shape from compile-time constants and the device
  profile's workgroup memory, within `MIN_SHARED_MEMORY_BYTES` unless the profile reports more.
- The ReLU epilogue and the pre-activation write keep working as today.
- Every rounded value stays the same: the products and partial sums are the same `OpFMul` and
  `OpFAdd` with `NoContraction`, in the same order, so the outputs equal today's bit for bit.
- DESIGN.md § 3.1 (workgroup sizes per kernel kind), § 5.2 (the M1 matmul now tiled) and the
  revision line; `CHANGELOG.md`.

## Scope

- `crates/modelspec-runtime/src/spirv/emit.rs` (`Module::matmul`, its body selection).
- `crates/modelspec-core/src/schedule.rs` and `crates/modelspec-core/src/ir/kernel.rs` where the
  workgroup size and count of a matmul kernel are set; `crates/modelspec-core/src/ir/verify.rs` if
  it checks them.
- `crates/modelspec-runtime/src/vulkan/dispatch.rs` if a dispatch needs a 2-D workgroup count.
- Tests: `crates/modelspec-runtime/tests/vulkan_*.rs`, `spirv_val.rs`, schedule tests and the
  `ir --stage kernel` snapshots of the CLI tests.
- `DESIGN.md`, `CHANGELOG.md`.

## Acceptance

The forward matmul is at most half its m1-e time, with the same results.

1. `train --backend vulkan --profile` on a 144,000-step copy of `examples/mnist_mlp.mspec`, 3 runs
   after a warm-up on the RTX 3090 with no other compute process: the median over the 3 runs of the
   `k0` site's mean µs is at most 9.0 (from 18.372 / 19.279 / 18.545).
2. The stdout of `train examples/mnist_mlp.mspec --backend vulkan` (1,800 steps) has the same
   sha256 over 5 runs, equal to the m1-e head's `b9bb028a…6904`.
3. Device tests: every matmul shape in the existing tests, plus non-multiples of the tile size in
   each of `m`, `k` and `n` and `k = 1`, equal the CPU oracle bit for bit where `negotiate` gives
   `bit-for-bit`, and within the per-operator tolerance of
   `approval-record:m1-tolerance-numerics-class` otherwise; the ReLU epilogue with and without the
   pre-activation write.
4. `spirv-val` passes on every emitted matmul module; a module whose workgroup memory exceeds the
   device profile's is refused before pipeline creation.
5. Every non-matmul dispatch of the step plan is unchanged: the `ir --stage kernel` diff against
   `7a488c1` touches only the 5 matmul kernels (`k0`, `k1`, `k6`, `k8`, `k11`), and no matmul
   site's profiled mean µs is above its m1-e value.
6. `task ci` exits 0.

## Out of scope

- Split-K, cooperative matrices, reading transposed operands strided, fusing other ops into the
  matmul.
- The weight-gradient threshold and the wave's closing re-time
  (`story:m1-f-weight-grad-matmul-tiled`).
