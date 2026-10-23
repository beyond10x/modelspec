---
format: aep.planning-md/3
id: story:cpu-matmul-output-skips-zero-fill
kind: story
status: draft
title: CPU matmul output is not zero-filled before the kernel writes it
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/matmul.rs
revision: 2
---
# CPU matmul output is not zero-filled before the kernel writes it

## Why

Review of 0.5.0 (finding P6), still open at main 496e98e. The CPU matmul resizes its output with
`T::ZERO` (`crates/modelspec-runtime/src/cpu/matmul.rs:117`) and then writes every element, so the
fill is a memset the result never reads.

Outcome: the CPU matmul writes each output element once.

## Acceptance

- **no-memset-in-profile**: a CPU profile of the MNIST example shows no memset attributed to the
  matmul output.
- **same-bits**: matmul outputs and the per-step loss are bit-identical to before, including the
  `m`, `k` or `n` of 0 cases (which must still produce zeros where `k` is 0).
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/cpu/matmul.rs`

## Out of scope

- The blocking and row order of the kernel.
