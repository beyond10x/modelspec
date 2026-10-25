---
title: Roadmap
description: Milestones M0 to M4 from DESIGN.md § 1. M0 is built; M1, GPU execution through Vulkan, is in progress and trains on the GPU since 0.4.0; the BERT operators of M3 run on the CPU reference.
---

# Roadmap

The milestones are those of [DESIGN.md § 1](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#1-scope-and-milestones).
M0 is built and released as 0.1.0. M1 is in progress; the latest release is 0.13.0. Some M2 items
were done early, in M1. The current milestone is a small BERT, pre-trained from scratch, which is
one of the two proof models of M3: its operators run on the CPU reference interpreter, and their
Vulkan kernels are planned. M4 is design.

## M0: semantic core

:::shipped[Built, released as 0.1.0]

A parser for the M0 subset, name resolution, the type and shape checker, typed SSA Tensor IR with
a verifier, dead-code elimination, reverse-mode autodiff, the SGD update as Training IR, the CPU
reference interpreter, and `modelspec check`, `ir` and `train --backend cpu`.
[Status](/docs/status) lists each capability with the test that holds it.

:::

## M1: native Vulkan execution

:::planned[In progress, partly released]

| Release | What M1 gained |
|---|---|
| 0.4.0 | `modelspec train --backend vulkan` trains both examples on one Vulkan device with one host wait per run, within the GPU tolerance of the CPU reference; `modelspec inspect-device --backend vulkan` reports each device; CPU checkpoints and `modelspec infer --backend cpu` |
| 0.6.0 | `modelspec compile` writes an artifact directory; `train --backend vulkan` writes and resumes checkpoints, waiting once more per requested checkpoint; `infer --backend vulkan` runs a checkpoint on the device |
| 0.7.0 | `train --backend vulkan` refuses a diverged run with `E-TRAIN-001` as the CPU run does; the artifact reader bounds the plan verifier's work on an untrusted artifact |
| 0.7.1 | `train --backend vulkan --profile` prints each dispatch's FLOP, bytes and achieved throughput; the Vulkan descriptor pool counts every view a dispatch reads and writes |
| 0.8.0 | `--stats` and `--profile` print the barriers and device allocations of a step and the peak device bytes of the run |
| 0.9.0 | a register-blocked matmul tile for large outputs; a matmul reads a transposed operand in place; the opt-in `--fast-math` |
| 0.10.0 | deterministic fused multiply-adds in every reduction, on the CPU and the GPU; scheduling for a device profile; elementwise chains as one dispatch; the driver's pipeline statistics under `--profile` |
| 0.11.0 | no change to the Vulkan backend; the observatory UI's layout, accessibility and screen-reading gates with a ratchet, in `task ui-ux` on every merge |
| 0.12.0 | no change to the Vulkan backend; AdamW trains on the CPU reference only |
| 0.13.0 | a `relu_grad` whose gradient operand is a matmul output nothing else reads is an epilogue of that matmul, and the `mse` loss sum shares a barrier interval with the largest matmul; the same bits, and the `mlp_sweep` b8192 h4096 step on the RTX 3090 at 1.044 of PyTorch eager (was 1.111) |

The 9 acceptance boxes of DESIGN.md § 9 are not ticked.

:::

M1 is the first milestone that meets the project's end-to-end goal: training on an NVIDIA RTX 3090
through Vulkan compute and SPIR-V, without the CUDA API or toolkit. It adds:

- a headless Rust Vulkan runtime (`ash`): device discovery that reports each capability it found,
  buffer allocation, upload and download, dispatch and synchronization;
- SPIR-V kernels built in Rust (`rspirv`) for the M0 operators: elementwise, two-stage reductions,
  transpose, a bounds-safe `f32` matmul, and the SGD update;
- the compiled forward pass, backward pass and update executed on the GPU, and held to the CPU
  reference within documented tolerances, including shapes not aligned to workgroup sizes;
- checkpoint save and load with the same parameter identities, and `compile`, `infer` and
  `inspect-device` on the command line.

The M1 acceptance criteria are the checklist in
[DESIGN.md § 9](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#m1-acceptance-criteria-unambiguous).
`modelspec-core` stays free of any Vulkan dependency; `task core-no-vulkan` fails if one appears.

## M2: optimizing compiler

DESIGN.md's M2 adds whole-step liveness, buffer reuse, elementwise fusion, a tiled shared-memory
matmul, a kernel scheduler, profiling and a kernel cache. Done early, in M1: elementwise fusion and
per-dispatch profiling (`train --profile`) in 0.4.0, the tiled shared-memory matmul in 0.5.0,
register blocking in 0.9.0, and fusion of whole elementwise chains in 0.10.0. The rest is design.

## M3: neural-network essentials

DESIGN.md's M3 adds `f16` where the device supports it, AdamW, softmax, layer norm, embeddings,
attention with causal and bidirectional masks, GELU, and two proof models: small GPT-like training
and small BERT-like masked-language-model pre-training. The BERT model is the current milestone.

:::shipped[On the CPU reference]

Released in 0.10.0: `embedding` with `i32` row ids, `add_rows`, `gelu`, `layer_norm`, `softmax`
and multi-head `attention` with a key mask, with `zeros()` and `ones()` initializers, and the
three commands that prepare a corpus, `modelspec data prepare`, `data tokenizer` and `data pack`.
Released in 0.11.0: `cross_entropy`, `dropout`, `train --data` on a packed directory, and
held-out evaluation during training with `--eval-every`. Each operator's gradient is checked
against central finite differences ([BERT operators](./concepts/bert-operators.md)).
Released in 0.12.0: `optimizer = adamw(...)`, AdamW with decoupled weight decay and a learning
rate that warms up linearly from 0 and decays linearly to 0 at the last step.

:::

:::caution[Planned]

Vulkan kernels for the row-wise BERT operators and then for
`attention`, `cross_entropy` and AdamW, and BERT-tiny pre-trained on the GPU until its held-out
loss is below the unigram baseline. `f16`, causal masks
and the GPT-like model are design.

:::

## M4: device specialization

:::planned[Not built]

The cooperative-matrix Tensor Core path where the driver exposes it, mixed precision with `f32`
accumulation, recompute planning and autotuning. Designed in DESIGN.md; not built.

:::

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
