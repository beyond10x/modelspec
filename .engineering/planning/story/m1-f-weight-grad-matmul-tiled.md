---
format: aep.planning-md/3
id: story:m1-f-weight-grad-matmul-tiled
kind: story
status: active
title: Tiled matmul tuned for the weight-gradient shape, and the closing re-time against release 0.5.0
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
- depends_on: story:m1-f-forward-matmul-tiled
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
revision: 14
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T15:28:57Z", actor: "human:timo", revision: 11}
- {from: "proposed", to: "active", at: "2026-10-22T15:28:58Z", actor: "human:timo", revision: 12}
---
# Tiled matmul tuned for the weight-gradient shape, and the closing re-time against release 0.5.0

## Context

Wave m1-f profile on `7a488c1` (see `story:m1-f-forward-matmul-tiled`): `k11` `433c82fb`, the dW1
matmul `[784,100] x [100,64]` with 784 workgroups, measured 6.697 / 7.218 / 6.957 µs mean over 3
profiled runs of 144,000 steps on the RTX 3090 at a mean P0 clock of 1951 MHz. The whole step is
40.225 µs by the ADR-007 method (wave m1-e close, median of 5 runs of 144,000 steps). The working
target of ADR-007 is 31.4 µs per step.

The weight-gradient shape has a short contracting index (`k = 100`) and a tall output (`m = 784`),
unlike the forward matmul (`k = 784`, `m = 100`), so the tile shape chosen for `k0` need not suit it.

## What it changes

- The tile and workgroup shape of a matmul kernel are chosen per kernel shape, with the
  weight-gradient shape getting its own choice where it measures faster.
- The same accumulation order as `story:m1-f-forward-matmul-tiled`: one invocation per output
  element over ascending `k`, no split-K, the same bits.
- DESIGN.md § 5.2 and the revision line if the shape rule changes; `CHANGELOG.md`.

## Scope

- `crates/modelspec-runtime/src/spirv/emit.rs` (`Module::matmul`), `crates/modelspec-core/src/schedule.rs`
  and `crates/modelspec-core/src/ir/kernel.rs` where the tile choice is made.
- Tests as in `story:m1-f-forward-matmul-tiled`.
- `DESIGN.md`, `CHANGELOG.md`.
- Shares every file with `story:m1-f-forward-matmul-tiled`; it runs after that story merges.

## Re-based, 2026-10-23 (wave m1-g)

Release 0.5.0 (`main` `7c45c1a`) carries `story:m1-f-forward-matmul-tiled`, whose tile already
applies to `k11`. Wave m1-g profile on `7c45c1a`, 3 runs of 144,000 steps after a warm-up, P0
1950.0 MHz per run:

- `k11` `d7d3124e`, 196 workgroups (`MatMulTile::of(784, 100, 64)`: 128 invocations, 16 columns,
  2 rows per invocation, depth 32): 4.911 / 4.912 / 4.905 µs.
- `k0` `5666286c`, 52 workgroups: 8.470 / 8.397 / 8.397 µs.
- Whole step by the ADR-007 method at release 0.5.0: 27.282 µs, median of 5 runs, so the former
  acceptance 5 (at most 31.4 µs) is met before this story starts. The acceptance below replaces it
  with a no-regression re-time against the release.

## Acceptance

1. `train --backend vulkan --profile`, 144,000 steps, 3 runs after a warm-up on the RTX 3090: the
   median over the 3 runs of the `k11` site's mean µs is at most 4.0 (from 4.911 / 4.912 / 4.905
   on `7c45c1a`), and the `k0` site stays at most 9.0.
2. The 1,800-step stdout sha256 over 5 runs equals `b9bb028a…6904`.
3. The device tests of `story:m1-f-forward-matmul-tiled` acceptance 3 pass for the weight-gradient
   tile choice too.
4. `task ci` exits 0.
5. Closing re-time: `7c45c1a` and the unit head timed in one session by the ADR-007 method, 5 runs
   of 144,000 steps each, median, range and mean P0 clock per run, at a mean P0 clock within 1935
   to 1965 MHz. The unit head's median is at most the base's median.
6. Miss rule: if acceptance 1 misses 4.0 after one fix round, the unit closes with the measured
   number when acceptances 2 to 5 hold, and the remaining gap becomes a new draft story.

## Out of scope

- Split-K, cooperative matrices, reading the transposed operand strided (dropping its transpose
  kernel), fusing other ops into the matmul.

