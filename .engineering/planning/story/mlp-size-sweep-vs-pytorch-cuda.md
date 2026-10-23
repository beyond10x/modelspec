---
format: aep.planning-md/3
id: story:mlp-size-sweep-vs-pytorch-cuda
kind: story
status: implemented
title: MLP size sweep against PyTorch CUDA on the RTX 3090
relations:
- decomposes: epic:transformer-vs-pytorch-cuda
- serves: vision:modelspec
scope:
- confidence: cited
  path: crates/modelspec-runtime/benches/mlp_sweep.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: opponents/mlp_sweep/
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-23T17:50:00Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-23T17:50:00Z", actor: "human:timo", revision: 5}
- {from: "active", to: "implemented", at: "2026-10-23T19:47:09Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"test_result":1}}}
---
# MLP size sweep against PyTorch CUDA on the RTX 3090

## Context

`epic:transformer-vs-pytorch-cuda`. The only GPU benchmark today is the 784 → 64 → 10 MLP at batch
100, about 20.45 MFLOP per step (`epic:pytorch-10x-mnist`). This story runs the same MLP shape
family at sizes where the GPU does real arithmetic, with the operators that already exist, so it
can start before any transformer operator lands. The epic's measurement rules apply.

## Scope of the sweep

- Hidden width 64, 256, 1024, 4096 and batch 100, 1000, 8192 (12 points), 784 inputs and 10
  outputs, MSE, SGD, `f32`, synthetic or MNIST data as the scoper settles.
- One `.mspec` per point, or one model whose sizes are set at compile time, as the scoper settles
  from what the language supports today.
- The ModelSpec side is a new Rust bench, `crates/modelspec-runtime/benches/mlp_sweep.rs`, with
  its `[[bench]]` entry in `crates/modelspec-runtime/Cargo.toml`. The PyTorch side (eager and
  `torch.compile`, CUDA, TF32 off) runs outside the repository.

## Acceptance

- `verification-report:mlp-size-sweep-<date>` holds, per point: FLOP per step from the shapes,
  ModelSpec Vulkan µs per step, PyTorch eager and `torch.compile` µs per step, each the median of
  3 runs, and GFLOP/s for all three from the same FLOP count.
- The report names the host, the driver, the PyTorch version, the ModelSpec and PyTorch commands,
  and the GPU clock during each run.
- For each of the three sides and each batch, the report names the smallest hidden width at which
  that side is no longer launch-bound: the first width whose µs per step is more than 10 % above
  the previous width's at that batch, or "not reached" when no width is. The crossover point the
  pitch slide shows is the one at batch 1000.
- Every point that ModelSpec cannot compile or run is listed with the refusal it printed.
