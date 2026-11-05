---
title: Why ModelSpec
sidebar_position: 1
description: What ModelSpec is for. One declared training program lets the compiler build a runtime for one target and train faster than a general-purpose framework; that speed is planned as a general result, and BERT-tiny against PyTorch on an RTX 3090 measures the way there, with a shorter step and a shorter time to the held-out baseline than torch.compile's in the series of 2026-11-05.
lede: "ModelSpec exists to train models faster than a general-purpose framework, by compiling one declared training program into a runtime built for one target, and that speed is the goal, measured on BERT-tiny and not yet shipped as a general result."
source: "The project's vision (north star, 2026-10-28); DESIGN.md § 0 and § 9; website/docs/benchmarks.md § Transformer block (release 0.21.0) and § BERT-tiny pre-training, time to the held-out baseline; crates/modelspec-core/src/autodiff.rs and schedule.rs; the Vulkan-against-CPU tests listed on the Status page."
---

# Why ModelSpec

ModelSpec exists to train models faster than a general-purpose framework can. It gets there by
compiling one declared training program into a runtime built for one target: one use case, one
operating system, one GPU chipset. Nothing in that runtime has to stay general while it runs.

:::caution[Planned]

The speed is the goal, not a shipped general result. It is measured on one model: pre-training
BERT-tiny at batch 32 on the RTX 3090, ModelSpec's step took a median of 3,697.1 µs against
4,109.6 µs on `torch.compile` and 7,518.0 µs on PyTorch eager in the series of 2026-11-05, 3 runs
per side with the clocks not locked. The transformer block below was last timed on 2026-10-27.
The figures are below and on the [Benchmarks](/docs/benchmarks) page.

:::

## 1. The compiler knows everything before it runs

A `.mspec` file declares the whole training step: the model, the loss, the optimizer, the data
bindings and the execution policy. The compiler checks it, derives the gradients and compiles
forward pass, loss, backward pass and update into one verified program
([the pipeline](./pipeline.md)). Every shape, every buffer and every dispatch of a step is known
before the first step runs.

A general-purpose framework is built to run any model on any supported device, and its runtime
carries that generality into every step. A ModelSpec runtime is compiled for one program on one
target, so it can leave the generality out and specialise the rest: kernel shapes, fusion, buffer
layout, scheduling for the device.

:::shipped[Today]

The whole step compiles into one Training IR program, and since 0.10.0 the kernel plan is
scheduled for the profile of the device it runs on. One target exists: Vulkan compute and SPIR-V
on one NVIDIA RTX 3090 on Linux.

:::

:::planned[Planned]

Runtimes built for other use cases, operating systems and GPU chipsets. "A chipset" means a
runtime specialised for the device profile the driver reports, through Vulkan and SPIR-V. It never
means NVIDIA SASS or the CUDA toolkit ([DESIGN.md § 0](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#0-executive-decision)).

:::

## 2. Optimizations that cannot change a result

ModelSpec is built spec first. The language and its semantics are specified in
[DESIGN.md](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md), and the records a run
writes (artifacts, checkpoints, training and benchmark runs) are an [ESS](https://beyond10x.github.io/ess/) ([GitHub](https://github.com/beyond10x/ess))
specification in `spec/`
from which their types are generated. The CPU reference interpreter runs the same verified IR as
the GPU backend. Every kernel, fusion and compiler pass is checked against that reference within stated
tolerances, and every gradient rule against finite differences.

That makes the compiler safe to optimize hard. A change, written by a person or by an agent, that
alters a result fails the gate and is refused. A change that only makes a step faster is kept. The
project can try aggressive optimizations quickly, because a wrong one cannot land unseen.

:::shipped[Today]

The Vulkan kernels are tested against the CPU interpreter on the same IR; the
[Status](/docs/status) page lists each capability with the test that holds it. Speed changes land
the same way, and their timed runs are recorded in `benchmarks/runs.yaml`.

:::

## 3. The worked example: BERT-tiny

BERT-tiny on one RTX 3090 against PyTorch on the same card is the measure. First the baseline,
then three steps.

The baseline for one part of it is released: the two encoder layers of BERT-tiny
(`examples/transformer_block.mspec`), at batch 8, measured on 2026-10-27 and published in 0.21.0
([Benchmarks § Transformer block](/docs/benchmarks#transformer-block)):

| Side | µs per step | Against ModelSpec |
|---|---|---|
| ModelSpec Vulkan | 93,868.4 | |
| PyTorch CUDA eager | 3,937.184 | 23.8× faster |
| PyTorch CUDA torch.compile | 1,971.859 | 47.6× faster |

On small models the ModelSpec step is already the shorter one: on the MNIST shapes of the MLP sweep (784 → 64 → 10, batch 100) a ModelSpec step takes 30.726 µs against
393.441 µs on PyTorch eager ([Benchmarks § Batch 100](/docs/benchmarks#batch-100)). The
transformer block is where the gap was on 2026-10-27; it has not been timed since.

The full model, `examples/bert_tiny.mspec` at batch 32 for 10,000 steps, was timed on 2026-11-05
with the build released as 0.26.0, 3 runs per side in turn
([Benchmarks § BERT-tiny pre-training](/docs/benchmarks#bert-tiny-pre-training-time-to-the-held-out-baseline)):

| Side | µs per step, median | µs per step, 3 runs | Training seconds to the held-out baseline, median | Training seconds to the baseline, 3 runs |
|---|---|---|---|---|
| ModelSpec Vulkan | 3,697.139 | 3,641.5, 3,697.1, 3,768.8 | 4.024 | 3.96, 4.02, 4.06 |
| PyTorch CUDA torch.compile | 4,109.567 | 4,096.0, 4,109.6, 4,118.4 | 4.504 | 4.50, 4.51, 4.43 |
| PyTorch CUDA eager | 7,517.975 | 7,593.0, 7,518.0, 7,461.4 | 8.252 | 8.44, 8.25, 8.18 |

Every ModelSpec run is under every `torch.compile` run on both figures, the nearest two 327.2 µs
(8.0 %) apart per step and 0.37 s (8.4 %) to the baseline. It is one series of 3 runs per side
with the clocks not locked; the series before it that day, at 1cf9e12f, overlapped to the
baseline.

The ladder from there:

| Step | Done when | Status |
|---|---|---|
| Baseline | Full BERT-tiny timed against PyTorch eager and `torch.compile` on the same RTX 3090, per step and to a held-out loss | block released in 0.21.0; full model timed per step and to the held-out baseline, published on [Benchmarks § BERT-tiny pre-training](/docs/benchmarks#bert-tiny-pre-training-time-to-the-held-out-baseline) |
| A | The transformer block within 3× of PyTorch eager, same day, same device and driver | planned |
| E | Full BERT-tiny at parity with PyTorch eager | measured in 0.26.0: the median step is 0.492 of eager's (2026-11-05) |
| F | Faster than `torch.compile`, per step and to the held-out target | measured in 0.26.0 in one series: median step 0.900 of `torch.compile`'s, median time to the target 0.893, every run under on both figures; the same over repeated series is planned |

Each step is a sequence of measured units: profile where the step's time goes, change the
largest site, re-run the CPU-reference tests, time it, publish the row. The
[Roadmap](../roadmap.md) keeps the order.

## Shipped and planned

| | Status |
|---|---|
| The whole training step as one declared, typed, compiled program | shipped |
| Training on one Vulkan GPU, held to the CPU reference | shipped, since 0.4.0 |
| Kernel plans scheduled for the device profile | shipped, since 0.10.0 |
| BERT-tiny pre-trained on the RTX 3090 ([run record](../runs/bert-tiny-gpu-pretraining.md)) | shipped, 2026-10-26 |
| The transformer block timed against PyTorch | shipped, 0.21.0 |
| A shorter step than PyTorch on the small MLPs of the sweep | measured, see [Benchmarks](/docs/benchmarks) |
| BERT-tiny faster than PyTorch eager and `torch.compile`, per step and to the held-out baseline | measured in one series of 3 runs per side, 0.26.0, see [Benchmarks](/docs/benchmarks) |
| Runtimes for other operating systems and GPU chipsets | planned |

What ModelSpec does not aim at: a PyTorch-compatible API, Python bindings, the CUDA toolkit, or
emitting SASS. The speed has to come from compiling one program for one target, not from a vendor
toolkit.
