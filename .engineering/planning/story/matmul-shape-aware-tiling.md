---
format: aep.planning-md/3
id: story:matmul-shape-aware-tiling
kind: story
status: active
title: Matmul tiling chosen by shape, deterministic
relations:
- depends_on: story:profile-begin-stamp-after-barrier
- blocks: story:weight-grad-matmul-structural
- serves: vision:modelspec
scope:
- confidence: inferred
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: website/docs/guides/inspect-a-run.md
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T10:10:23Z", actor: "human:timo", revision: 8, decided_on: {"recorded":{"review_outcome":6}}}
- {from: "proposed", to: "active", at: "2026-10-24T10:10:23Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"review_outcome":6}}}
---
# Matmul tiling chosen by shape, deterministic

## Why

Review of 0.7.1 (finding 3, its deterministic part) and Review of 0.5.0 (finding P7), at main
496e98e. The matmul tile fixes a workgroup of 128 and clamps the output columns per invocation to 16
whatever the shape (`crates/modelspec-core/src/ir/kernel.rs:195-210`). The workgroup strides in
shared memory are unpadded and unmeasured (`crates/modelspec-runtime/src/spirv/emit.rs:1391`,
`:1555-1573`, `:1567-1571`), so large matmuls may hit bank conflicts.

A tile is printed as `tile … cols rows depth` and kernel ids are derived from it
(`crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir:25`), so the kernel golden and the kernel
lines the guide quotes change with it. DESIGN.md states the tile rule this story changes ("128 for
every shape today", DESIGN.md:164, and the matmul tiling at DESIGN.md:275), so DESIGN.md changes
with it and its revision line is bumped.

Outcome: the matmul tile, including shared-memory padding, depends on `m`, `k` and `n`, and large
shapes run closer to the device's throughput, without changing a bit on the device.

## Acceptance

- **baseline-at-head**: before the change, the story measures on the RTX 3090, by the GPU timing
  method of ADR-007 (DESIGN.md § 12), the `mlp_sweep` median step time at b8192 h4096 and per-matmul
  GFLOP/s from `--profile` for every matmul site of that point and of the MNIST example, at its
  starting commit. The 0.6.0 figure (31067.1 µs, b2f9553) is context only.
- **sweep-median**: after the change, the median step time at b8192 h4096 is below the
  baseline-at-head figure; both are reported.
- **per-matmul-gflops**: per-matmul GFLOP/s after the change is reported beside the baseline for
  every site.
- **padded-strides-measured**: a padded shared-memory stride is kept only when the MNIST matmul's
  `mean_us` from `--profile` falls by at least 3 % against the unpadded stride on the same run;
  otherwise the story records both figures and keeps the unpadded stride.
- **same-bits**: matmul outputs are bit-identical to the plan before the change on the same device;
  CPU equality is the § 5.4 class the device reports (the RTX 3090 reports
  `shaderDenormPreserveFloat32` false). One invocation accumulates in ascending `k`, and the
  contracting index is not split (no split-K, DESIGN.md:281).
- **design-states-tile-rule**: DESIGN.md:164 and the matmul tiling at DESIGN.md:275 state the
  shape-dependent tile rule, and the revision line is bumped.
- **goldens-regenerated**: `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir` and the kernel
  lines in `website/docs/guides/inspect-a-run.md` are regenerated in the same change, and
  `crates/modelspec-cli/tests/ir_kernel.rs` and
  `crates/modelspec-cli/tests/adversary_guide_kernel_lines.rs` pass.
- **register-blocked**: for a shape whose output has at least 1024 rows and 1024 columns, each
  invocation computes a block of `rows` by `cols_per_invocation` output elements (at least 4 by 4)
  from register-held operand fragments staged through workgroup memory, each element still
  accumulated by one invocation over ascending `k` from the `k = 0` product; the tile rule
  prints the block and the kernel signature changes with it. Added 2026-10-24 by the wave
  coordinator for the goal "modelspec beats python+cuda in the bigger batch benchmark": one or two
  output elements per invocation bound the kernel at 3.45 TFLOP/s at b8192 h4096 (31,067.1 µs,
  `benchmarks/runs.yaml` sweep-b8192-h4096-modelspec-vulkan) against 14.7 TFLOP/s of PyTorch eager
  CUDA (7,290.5 µs, sweep-b8192-h4096-pytorch-cuda-eager, TF32 off).
- **gap-to-pytorch-reported**: the `mlp_sweep` medians at b8192 for every hidden size and at b1000
  h1024 and h4096 after the change are reported beside the PyTorch eager CUDA medians recorded in
  `benchmarks/runs.yaml` (b8192: 506.9, 624.6, 1,932.1, 7,290.5 µs; b1000 h1024 555.1, h4096
  994.7 µs), with the ratio per point; this is reported, not gated, and the goal is checked at the
  release benchmark rerun.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`
- `website/docs/guides/inspect-a-run.md`
- `DESIGN.md` (DESIGN.md:164, :275, revision line)

## Out of scope

- FMA contraction and cooperative matrices, which story:fast-math-mode holds.
- The structural k11 change in story:weight-grad-matmul-structural, which this story blocks: that
  story's k11 figure and default-tile baseline are read after this re-tiling.
- Hardware-aware tile selection and autotune (DESIGN.md § 12, "Autotune storage").
