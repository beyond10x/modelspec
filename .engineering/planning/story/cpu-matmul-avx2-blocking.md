---
format: aep.planning-md/3
id: story:cpu-matmul-avx2-blocking
kind: story
status: active
title: CPU matmul with AVX2 and register blocking, with the same bits
relations:
- serves: vision:modelspec
- depends_on: story:cpu-matmul-row-order
- decomposes: epic:pytorch-10x-mnist
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: crates/modelspec-runtime/benches
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/cpu/matmul.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/cpu/mod.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/cpu/tensor.rs
- confidence: cited
  path: crates/modelspec-runtime/tests
revision: 11
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T02:05:23Z", actor: "human:timo", revision: 10}
- {from: "proposed", to: "active", at: "2026-10-22T02:05:23Z", actor: "human:timo", revision: 11}
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

Derived 2026-10-22 by `story-scoper`. Each line is **cited** (read from the story or the tree) or
**inferred**.

- Primary surface: `crates/modelspec-runtime`, the `cpu` module (cited).
- `crates/modelspec-runtime/src/cpu/interpreter.rs`: `matmul` (line 317), `transpose` (340),
  `eval_node` (223–272), `run` (115–129), `evaluate` / `evaluate_all` (75, 85) (cited).
- A new kernel module `crates/modelspec-runtime/src/cpu/matmul.rs`, declared in
  `crates/modelspec-runtime/src/cpu/mod.rs` (inferred; the story says "a new module beside it").
- `crates/modelspec-runtime/src/cpu/tensor.rs` (`Element`, lines 15–36): the kernel is generic over
  `T: Element`, so f32 and f64 reach AVX2 through a trait hook or a type dispatch (inferred).
- Tests: `crates/modelspec-runtime/tests/matmul_row_order.rs` holds the 0.1.1 reference loop and the
  seeded cases; a new test or an extension of it, with a hook that forces the portable path
  (cited for the file, inferred for the hook).
- `crates/modelspec-runtime/benches/mnist_shape.rs` (cited); `CHANGELOG.md` `[Unreleased]` (cited).
- Not touched: `modelspec-core`. `ops.rs:238` and `:242` emit `Transpose` → `MatMul`; the
  interpreter can read the transposed operand in place from the operand's defining node, and
  DESIGN.md § 3.1 leaves layout to the backend (cited), so no IR or design change (inferred).
- Not touched: CLI goldens, `website/docs/**` (no CPU timing stated), workspace `Cargo.toml` (cited).
- Constraint: `evaluate_all` promises every node's value (`interpreter.rs:62-67`, pinned by
  `tests/cpu_golden.rs:457-460`), so only `evaluate` (the `sgd_step` path) may skip the transpose
  copy (cited).
- Constraint: edition 2024 and `task lint` with `-D warnings` make `unsafe_op_in_unsafe_fn` an
  error, so intrinsics need explicit `unsafe {}` blocks; the runtime has no `unsafe` today and no
  lint forbids it (cited for the lint table, inferred for the gate effect).
- Confidence: high.

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
