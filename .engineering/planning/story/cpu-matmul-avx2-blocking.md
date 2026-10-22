---
format: aep.planning-md/3
id: story:cpu-matmul-avx2-blocking
kind: story
status: implemented
title: CPU matmul with AVX2 and register blocking, with the same bits
relations:
- serves: vision:modelspec
- depends_on: story:cpu-matmul-row-order
- decomposes: epic:pytorch-10x-mnist
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/matmul.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/tensor.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_avx2_counts.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_avx2_kernel.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/matmul_row_order.rs
revision: 15
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T02:05:23Z", actor: "human:timo", revision: 10}
- {from: "proposed", to: "active", at: "2026-10-22T02:05:23Z", actor: "human:timo", revision: 11}
- {from: "active", to: "implemented", at: "2026-10-22T02:57:26Z", actor: "human:timo", revision: 15, decided_on: {"recorded":{"test_result":1,"metric_observation":1,"review_outcome":2,"verification":1}}}
---
# CPU `matmul` with AVX2 and register blocking, with the same bits

## Context

`story:cpu-matmul-row-order` brought the 784 → 64 → 10 benchmark shape from 18.6 to 1.226 ms/step
(median of 7 interleaved pairs on one core, 15.2×). PyTorch on one CPU thread takes 526 µs/step
(`epic:pytorch-10x-mnist`), 2.33× faster. The profile after that change puts 71.9% of the time in
the `matmul` row loop, compiled to 4-lane SSE (`mulps`/`addps`, the generic x86-64 target), and
10.1% in the transpose copy that feeds `matmul` a transposed operand.

The oracle's bits may not change (DESIGN.md § 2: ascending `k` per output element, first term
first, no fused multiply-add). Wider lanes and blocking over `i` and `j` keep that order: each lane
holds one output element, and IEEE multiply and add round the same way in a vector lane as in a
scalar. Rust never contracts a multiply and an add into an FMA on its own.

## What it adds

- An AVX2 path for the f32 and f64 `matmul` row loop, compiled with `#[target_feature(enable =
  "avx2")]` and chosen at run time with `is_x86_feature_detected!`; the current loop stays the
  path for every other CPU and target.
- A register block over output rows and columns, held across the whole `k` loop, so each output
  element is loaded and stored once per `matmul`.
- `matmul` reads a transposed operand in place where the interpreter now copies it first.

## Scope

Derived 2026-10-22 by `story-scoper`; rewritten at the close of wave `cpu-perf-2` from the diff of
`f250dd5` against `cfb771b` (8 files, +1,174 / −40). Every line is now **cited** from that diff.

- `crates/modelspec-runtime/src/cpu/matmul.rs` (new, declared in `cpu/mod.rs`): public `Kernel`,
  `Lhs` and `matmul`; f32x8 and f64x4 AVX2 kernels with register blocks and runtime detection; the
  portable i-k-j loop; `element_counts`, which checks `m*k`, `k*n` and `m*n` with `checked_mul`.
- `crates/modelspec-runtime/src/cpu/tensor.rs`: the hidden `Element::matmul_avx2` hook.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`: a `Transpose` left operand of `matmul` is
  read in place; `evaluate` borrows inputs and parameters; `evaluate_all` still copies every node.
- `crates/modelspec-runtime/tests/matmul_row_order.rs` (extended),
  `tests/adversary_avx2_kernel.rs` and `tests/adversary_avx2_counts.rs` (new, from the adversary).
- `CHANGELOG.md` `[Unreleased]`, one line under `### Changed`.
- Not touched, as scoped: `modelspec-core`, CLI goldens, `website/docs/**`, `Cargo.toml`.
- Not touched, against the scoper's estimate: `benches/mnist_shape.rs` (the bench needed no change).
- Left for later: the right-operand transpose is still copied; no blocking over `k`.

## Acceptance

All CPU outputs are bit-identical to release 0.1.1, and the benchmark shape runs at or under
600 µs/step on the host and thread of the cpu-matmul-row-order measurement.

1. Bits unchanged: every CLI golden under `crates/modelspec-cli/tests/golden/` and every
   `train_linear` case pass unchanged; the benchmark prints weights blake3 `805f64bd…02df` and last
   loss bits `0x3dbbe63b`.
2. Bit equality on random shapes: the AVX2 path and the portable path agree with the 0.1.1 loop on
   at least 200 seeded shapes up to `[64, 784] x [784, 64]`, including shapes whose sizes are not a
   multiple of the block, `-0.0`, infinities and finite inputs, f32 and f64, by bit pattern. NaN
   inputs compare as NaN only (Rust does not fix the payload of an arithmetic NaN).
3. Speed: base and change binaries run interleaved on one pinned core, at least 5 pairs; the change's
   median is at or under 600 µs/step and both runs' output is recorded in the story's evidence.
   600 µs is the coordinator's estimate from the profile, not a measurement; a shortfall is recorded
   and the operator decides.
4. Planted defects: an FMA in the AVX2 kernel (`_mm256_fmadd_ps`) fails scenario 2; a block that
   drops the last partial column fails scenario 2 on a non-multiple shape.

## Out of scope

- Threads, FMA, `fast_math`, AVX-512, and building the crate with `target-cpu=native`.
- The GPU path, which is M1.
