---
format: aep.planning-md/3
id: story:vulkan-column-sums-coalesced
kind: story
status: active
title: Column sums read their rows coalesced
summary: LayerNormGradGamma, SumRows and the other column sums keep their ascending order and bits and read rows coalesced.
relations:
- decomposes: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
scope:
- confidence: inferred
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: inferred
  path: crates/modelspec-core/src/schedule.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T03:20:13Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T03:20:13Z", actor: "human:timo", revision: 3}
---
# Column sums read their rows coalesced

## Context

Profile 2026-10-29, perf/bert-step-tenth 9e448ada, `train examples/bert_tiny.mspec` 200 steps, packed-128, `--backend vulkan --profile`: steady 15.0 ms per step, 12.3 ms of it in kernels, 593 dispatches, 418 barriers and one submit and host wait per step. PyTorch eager 9.9 ms, torch.compile 4.9 ms (verification report of story:bert-tiny-vs-pytorch-cuda, training clock).

`LayerNormGradGamma` (1.5 ms) and `SumRows` (0.7 ms) take 2.2 ms per step together: each sums 4,096 rows of one of 128 columns serially in one invocation, reading with a stride of the row width, so 128 invocations run on a device of 10,496 lanes.

## Outcome

The column-sum kernels (LayerNormGradGamma, LayerNormGradBeta where separate, SumRows and every other row-reduction the BERT step dispatches) keep each column's ascending-row addition order, so the bits stay, and read their rows coalesced: a workgroup loads a tile of rows into workgroup memory and each column's invocation adds its values in order. If a probe shows that keeping the order cannot beat 0.3 ms, the lane stops and reports the figure; a change of summation order is a DESIGN.md numerics change and is not made here.

## Acceptance

- a probe first, with the kernel time against 2.2 ms;
- the bit tests of the BERT step unchanged; a device test of a [4097, 129] column sum against the CPU bit for bit, red first;
- interleaved A/B, 3 pairs, ratio in the commit.
