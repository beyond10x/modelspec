---
format: aep.planning-md/3
id: epic:transformer-vs-pytorch-cuda
kind: epic
status: draft
title: A transformer benchmark against PyTorch on CUDA, after an MLP size sweep
summary: BERT-tiny and its encoder block trained by ModelSpec on Vulkan and by PyTorch on CUDA on the RTX 3090, compared on step time, tokens per second, time to held-out loss and achieved GFLOP/s
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: epic:bert-from-scratch
- informed_by: epic:pytorch-10x-mnist
revision: 2
---
# A transformer benchmark against PyTorch on CUDA

## Outcome

A real model, a small transformer written in `.mspec`, trains on the RTX 3090 through
`modelspec train --backend vulkan`, and the same model trains in PyTorch on CUDA on the same host
with the same initialisation, data order and step count. The repository publishes the comparison:
step time, tokens per second, wall time to a held-out loss target, and how much of the GPU each
side used. Before the transformer exists, a size sweep of the existing MLP shows where each side
stops being bound by launch overhead and starts being bound by arithmetic.

## Operator intent

"i think relatively early or next in our roadmap we should do an actual real-life example +
benchmarp + competition with python + cuda. maybe a small transformer? we are not yet maxing out
the gpu inn our tests..." (operator, 2026-10-23).

## Why this is not the MNIST epic

`epic:pytorch-10x-mnist` gates one shape: 784 → 64 → 10 at batch 100, about 20.45 MFLOP per step
(its own body). At that size the PyTorch step is host-side launch cost (inferred there, not
profiled) and the GPU does little arithmetic. This epic measures the regime where the GPU is busy:
larger matrices first, then attention.

## Why the transformer is BERT-tiny

`epic:bert-from-scratch` already plans the operators, the data pipeline and a GPU pre-training run
of BERT-tiny (2 layers, hidden 128, 2 heads, sequence 128). This epic reuses that model and adds
no second transformer.

## Order

- Early, with the operators that exist today: `story:profile-achieved-throughput`,
  `story:mlp-size-sweep-vs-pytorch-cuda` and `story:benchmark-results-page`.
- After the BERT-tiny CPU model and its Vulkan kernels: `story:transformer-block-benchmark`. Its
  model needs no tokenizer, corpus or MLM batches, but the declared edges of
  `story:bert-vulkan-kernels-rowwise` (shared files, that story's body) order it after
  `story:bert-tiny-cpu-model`, and therefore after the BERT data pipeline.
- Last: `story:bert-tiny-vs-pytorch-cuda`, after `story:bert-tiny-gpu-pretraining`.

## In scope

- An MLP size sweep on the Vulkan backend and on PyTorch CUDA (eager and `torch.compile`).
- Per-dispatch FLOP and byte counts from the kernel plan, so `train --profile` reports achieved
  GFLOP/s and GB/s, the measure of how much of the GPU a run uses. The PyTorch side's GFLOP/s is
  the same FLOP count divided by its step time.
- A transformer-block benchmark on synthetic data: forward, backward and SGD update.
- BERT-tiny MLM pre-training to a held-out loss target, wall time compared with PyTorch.
- A results page on the website and a pitch-deck slide, every figure from a verification report.

## Out of scope

- Any Python, PyTorch or CUDA in the repository (AGENTS.md:117). The PyTorch side, its data
  export and its plots stay outside the repository, as for `verification-report:mnist-benchmark-cuda-baseline`;
  the repository holds the ModelSpec side and the figures as verification reports.
- A speed gate. This epic measures and publishes; a gate like ADR-007 is a later decision.
- Mixed precision, `f16` and tensor cores (DESIGN.md § 1 row M4). PyTorch runs `f32` with TF32 off,
  as in the MNIST baseline.

## Acceptance

- A verification report per benchmark names the host, driver, PyTorch version, both commands, the
  median of at least 3 runs, and the GPU clock during each run.
- `train --profile` prints achieved GFLOP/s and GB/s per dispatch site and per step.
- The results page and the pitch deck cite each figure to its verification report.

## Measurement rules

Every story of this epic follows these rules.

- Timing as in `epic:pytorch-10x-mnist` § Measurement: the training loop, excluding setup,
  compilation, warmup, evaluation and IO; median of at least 3 runs.
- A timed run has the GPU to itself (no other compute process in `nvidia-smi`); clocks are not
  locked.
- Device runs are the operator's to start.

## Dependencies

- `epic:m1-vulkan-execution` for the Vulkan runtime.
- `epic:bert-from-scratch` for the transformer operators, kernels and the BERT-tiny model.
