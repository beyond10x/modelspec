---
title: Why ModelSpec
sidebar_position: 1
description: What ModelSpec is for. One declared training program lets the compiler build a runtime for one target and train faster than a general-purpose framework; that speed is planned, and BERT-tiny against PyTorch on an RTX 3090 measures the way there.
lede: "ModelSpec exists to train models faster than a general-purpose framework, by compiling one declared training program into a runtime built for one target, and that speed is the goal, not yet shipped."
source: "The project's vision (north star, 2026-10-28); DESIGN.md § 0 and § 9; website/docs/benchmarks.md § Transformer block (release 0.21.0); crates/modelspec-core/src/autodiff.rs and schedule.rs; the Vulkan-against-CPU tests listed on the Status page."
---

# Why ModelSpec

ModelSpec exists to train models faster than a general-purpose framework can. It gets there by
compiling one declared training program into a runtime built for one target: one use case, one
operating system, one GPU chipset. Nothing in that runtime has to stay general while it runs.

:::caution[Planned]

The speed is the goal, not a shipped result. On a transformer, today's ModelSpec is many times
slower than PyTorch on the same GPU; the figures are below and on the
[Benchmarks](/docs/benchmarks) page.

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
transformer block is where the gap is, and closing it is the work.

The ladder from there:

| Step | Done when | Status |
|---|---|---|
| Baseline | Full BERT-tiny timed against PyTorch eager and `torch.compile` on the same RTX 3090, per step and to a held-out loss | block released in 0.21.0; full model in progress |
| A | The transformer block within 3× of PyTorch eager, same day, same device and driver | planned |
| E | Full BERT-tiny at parity with PyTorch eager | planned |
| F | Faster than `torch.compile`, per step and to the held-out target | planned |

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
| BERT-tiny faster than PyTorch eager and `torch.compile` | planned |
| Runtimes for other operating systems and GPU chipsets | planned |

What ModelSpec does not aim at: a PyTorch-compatible API, Python bindings, the CUDA toolkit, or
emitting SASS. The speed has to come from compiling one program for one target, not from a vendor
toolkit.
