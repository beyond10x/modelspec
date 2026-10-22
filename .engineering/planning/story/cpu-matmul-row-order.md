---
format: aep.planning-md/3
id: story:cpu-matmul-row-order
kind: story
status: implemented
title: CPU matmul in row order, with the same bits
relations:
- serves: vision:modelspec
- depends_on: epic:m0-semantic-core
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: crates/modelspec-runtime/Cargo.toml
- confidence: cited
  path: crates/modelspec-runtime/benches
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/tests
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T20:50:24Z", actor: "human:timo", revision: 7}
- {from: "proposed", to: "active", at: "2026-10-21T20:50:24Z", actor: "human:timo", revision: 8}
- {from: "active", to: "implemented", at: "2026-10-22T01:39:54Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# CPU `matmul` in row order, with the same bits

## Context

The MNIST benchmark of release 0.1.1 (`verification-report:mnist-benchmark-0-1-1`) trained a
784 → 64 → 10 MLP at 5,024 images/s on one thread; PyTorch eager reached 190,013, 37.8× faster,
with the same test accuracy and a relative L2 difference of at most 1.6e-7. The CPU `matmul`
(`crates/modelspec-runtime/src/cpu/interpreter.rs:313`) loops `i`, `j`, `k` and reads
`b[p * n + j]`, a stride of `n` elements per term.

The CPU interpreter is the oracle (DESIGN.md § 9), so its bits may not change: DESIGN.md § 2 fixes
the reduction order (each output element sums its `K` products in ascending `k`, starting from the
first term, no fused multiply-add). Loop order `i`, `k`, `j` with a row accumulator keeps that order
per element: the row starts as the products for `k = 0` and each later `k` adds its product.

## What it adds

- `matmul` in loop order `i`, `k`, `j`, the accumulator row initialised from the `k = 0` products
  (not from zero, so a `-0.0` first term stays `-0.0`).
- A Rust benchmark target (no Python), `cargo bench` or a `#[ignore]` timing test, that trains the
  benchmark's 784 → 64 → 10 shape on synthetic data with batch 100 for a fixed number of steps and
  prints images/s. It is not part of `task ci`.

## Scope

- `crates/modelspec-runtime/src/cpu/interpreter.rs`, a test under `crates/modelspec-runtime/tests/`,
  the benchmark target under `crates/modelspec-runtime/benches/` (and its `Cargo.toml` entry),
  `CHANGELOG.md`.

## Acceptance

All CPU outputs are bit-identical to release 0.1.1, and the benchmark shape trains at least 3× as
many images per second as 0.1.1 on the same host and thread.

1. Bits unchanged: every CLI golden under `crates/modelspec-cli/tests/golden/` and every
   `train_linear` case pass unchanged.
2. Bit equality on random shapes: a test compares the new `matmul` with the 0.1.1 loop (kept in the
   test) on at least 200 seeded shapes up to `[64, 784] x [784, 64]`, including `-0.0`, infinities
   and NaN inputs, f32 and f64, by bit pattern.
3. Speed: the benchmark target run on 0.1.1 and on the change, same host, 1 thread, 3 repeats each;
   the median ratio is at least 3. Both runs' output is recorded in the story's verification
   evidence. The 3× target is the coordinator's estimate, not a measurement; if the measured ratio
   falls short, the story records it and the operator decides.
4. Planted defect: initialising the accumulator row from zero fails scenario 2 on a `-0.0` case.

## Out of scope

- Threads, SIMD intrinsics, blocking for cache beyond the loop order, and `fast_math`.
- Comparison with PyTorch inside the repository.
