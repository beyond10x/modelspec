---
title: Roadmap
description: Milestones M0 to M4 from DESIGN.md § 1. M0 is built; GPU execution starts at M1 and is not built.
---

# Roadmap

The milestones are those of [DESIGN.md § 1](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#1-scope-and-milestones).
M0 is built and released as 0.1.0. Everything below it is design, not software.

## M0: semantic core

:::shipped[Built, released as 0.1.0]

A parser for the M0 subset, name resolution, the type and shape checker, typed SSA Tensor IR with
a verifier, dead-code elimination, reverse-mode autodiff, the SGD update as Training IR, the CPU
reference interpreter, and `modelspec check`, `ir` and `train --backend cpu`.
[Status](./status.mdx) lists each capability with the test that holds it.

:::

## M1: native Vulkan execution

:::planned[In progress, not released]

On `main`, not in a release yet: `modelspec train --backend vulkan` trains both examples on one
Vulkan device with one host wait per run, within the GPU tolerance of the CPU reference, and
`modelspec inspect-device --backend vulkan` reports each device. `modelspec compile` writes an
artifact directory, `modelspec train` writes and resumes checkpoints on both backends, the device
run waiting once more per requested checkpoint, and `modelspec infer` runs a checkpoint on the CPU
or on the device. The work is planned as `epic:m1-vulkan-execution` in the repository's planning
store.

:::

M1 is the first milestone that meets the project's end-to-end goal: training on an NVIDIA RTX 3090
through Vulkan compute and SPIR-V, without the CUDA API or toolkit. It adds:

- a headless Rust Vulkan runtime (`ash`): device discovery that reports each capability it found,
  buffer allocation, upload and download, dispatch and synchronization;
- SPIR-V kernels built in Rust (`rspirv`) for the M0 operators: elementwise, two-stage reductions,
  transpose, a bounds-safe `f32` matmul that is not claimed to be fast, and the SGD update;
- the compiled forward pass, backward pass and update executed on the GPU, and held to the CPU
  reference within documented tolerances, including shapes not aligned to workgroup sizes;
- checkpoint save and load with the same parameter identities, and `compile`, `infer` and
  `inspect-device` on the command line.

The M1 acceptance criteria are the checklist in
[DESIGN.md § 9](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#m1-acceptance-criteria-unambiguous).
`modelspec-core` stays free of any Vulkan dependency; `task core-no-vulkan` fails if one appears.

## M2 to M4

:::planned[Not built]

Designed in DESIGN.md; no code and no stories yet.

:::

| Milestone | Adds |
|---|---|
| M2: optimizing compiler | whole-step liveness, buffer reuse, elementwise fusion, tiled shared-memory matmul, a kernel scheduler, profiling and a kernel cache |
| M3: neural-network essentials | `f16` where the device supports it, AdamW, softmax, layer norm, embeddings, attention with causal masks, small GPT-like training |
| M4: device specialization | the cooperative-matrix Tensor Core path where the driver exposes it, mixed precision with `f32` accumulation, recompute planning, autotuning |

The language grows alongside: modules, `layer` instances and generic dimensions (the syntax in
[Syntax beyond M0](./examples/beyond-m0.md)) arrive with iteration 3 of DESIGN.md § 11, together
with the M2 work.

## Excluded for now

[DESIGN.md § 1](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#initial-deliberate-exclusions)
excludes these from the M0 and M1 design, and they may be considered later:

- distributed training and automatic cross-device partitioning;
- dynamic Python-like objects and implicit eager execution;
- user-authored GPU assembly and direct SASS emission;
- arbitrary runtime recursion, variable-length control flow and graph mutation during execution;
- arbitrary-stride views;
- a production data-loader ecosystem.

[DESIGN.md § 2.4](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#24-control-flow-and-dynamism)
also keeps runtime data-dependent `if` and `while` and dynamically ragged tensors out of scope. An
optional PTX backend is an open question in DESIGN.md § 12 and would not meet the no-CUDA rule.
