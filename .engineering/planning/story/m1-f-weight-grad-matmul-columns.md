---
format: aep.planning-md/3
id: story:m1-f-weight-grad-matmul-columns
kind: story
status: archived
title: 'Weight-gradient matmul at most 4.0 µs: several output columns per invocation'
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir
- confidence: inferred
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: inferred
  path: crates/modelspec-core/tests/adversary_kernel_two_row.rs
- confidence: inferred
  path: crates/modelspec-core/tests/kernel_plan.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/adversary3_vulkan_matmul_tiled.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_matmul_tiled.rs
- confidence: inferred
  path: website/docs/guides/inspect-a-run.md
revision: 16
transitions:
- {from: "draft", to: "proposed", at: "2026-10-23T08:16:30Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-23T08:16:30Z", actor: "human:timo", revision: 3}
- {from: "active", to: "archived", at: "2026-10-23T09:09:50Z", actor: "human:timo", revision: 16}
---
# Weight-gradient matmul at most 4.0 µs: several output columns per invocation

## Context

`story:m1-f-weight-grad-matmul-tiled` closed under its miss rule (acceptance 6). Wave m1-g profile
on the RTX 3090, 3 runs of 144,000 steps after a warm-up, P0 1965.0 MHz per run: the `k11` site,
the dW1 matmul `[784,100] x [100,64]` with 196 workgroups, measured 4.585 / 4.582 / 4.578 µs after
that story, from 4.911 / 4.912 / 4.905 µs at release 0.5.0. Of 26 tile-shape variants measured for
`k11` (workgroup 64 to 128, 8 to 64 columns, 1 to 4 rows, depth 8 to 100, 1 to 8 ahead, odd `b`
padding), none came below 4.6 µs. The implementor's reading, not measured: reaching 4.0 µs needs
a change to the kernel's structure rather than its tile shape, for example several output columns
per invocation.

## What it changes

- A matmul invocation may compute more than one output column, each with its own accumulation over
  ascending `k` from the `k = 0` product, so the bits stay those of the CPU oracle.
- The rule that picks it is per kernel shape, as `MatMulTile::of` is today.
- `Kernel::signature` covers every tile parameter (depth, ahead, columns per invocation), so a
  profile line names a different id when the module changes (adversary note on
  `crates/modelspec-core/src/ir/kernel.rs:359`).

## Scope

- `crates/modelspec-core/src/ir/kernel.rs`, `crates/modelspec-runtime/src/spirv/emit.rs`, the
  matmul tests in both crates, `DESIGN.md` § 5.2 and its revision line, `CHANGELOG.md`.

## Acceptance

1. `train --backend vulkan --profile`, 144,000 steps, 3 runs after a warm-up on the RTX 3090: the
   median of the `k11` site's mean µs is at most 4.0, and `k0` stays at most 9.0.
2. The 1,800-step stdout sha256 over 5 runs equals `b9bb028a…6904`.
3. Device tests bit-equal to the CPU at every new columns-per-invocation choice, at `k` = 1, at a
   partial last column block and at every tile remainder, with the validation layer on.
4. `task ci` exits 0.
5. The base and the head timed in one session by the ADR-007 method, 5 runs of 144,000 steps; the
   head's median is at most the base's.

## Out of scope

- Split-K, cooperative matrices, dropping the transpose kernel.
