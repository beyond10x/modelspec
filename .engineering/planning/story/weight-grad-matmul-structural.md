---
format: aep.planning-md/3
id: story:weight-grad-matmul-structural
kind: story
status: draft
title: 'Weight-gradient matmul at most 4.0 µs: a structural change past the tile'
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
- informed_by: verification-report:k11-multicol-2026-10-23
revision: 1
---
# Weight-gradient matmul at most 4.0 µs: a structural change past the tile

## Context

`verification-report:k11-multicol-2026-10-23` measured k11 (dW1, `[784,100] x [100,64]`) at a
median of 4.716 µs on main 24e84d0. Several output columns per invocation made it slower in every
variant tried. Probes put arithmetic at about 0.35 µs and the global fetch and barrier chain at
about 1.35 to 2 µs. The candidates left are split-K, reading the transposed operand strided
(dropping the transpose kernel), and cooperative matrices. None of them has been measured.

## What it decides first

- A prototype of each candidate is measured on k11 against the default tile in the same session,
  before any test is written. Only the fastest candidate that keeps the CPU oracle's bits is built.
  Split-K changes the order of accumulation, so it must keep ascending `k` from the `k = 0`
  product, or be refused for that reason.

## Acceptance

1. `train --backend vulkan --profile`, 144,000 steps, 3 runs after a warm-up on the RTX 3090: the
   median of the k11 site's mean µs is at most 4.0, and k0 stays at most 9.0.
2. The 1,800-step stdout sha256 over 5 runs equals `b9bb028a…6904`.
3. `task ci` exits 0.
4. Miss rule: if no candidate reaches 4.0 µs, the story closes with a verification report of every
   candidate's figure, and k11 keeps the default tile.
