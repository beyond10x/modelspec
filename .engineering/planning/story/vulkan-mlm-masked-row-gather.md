---
format: aep.planning-md/3
id: story:vulkan-mlm-masked-row-gather
kind: story
status: implemented
title: MLM head and loss over the masked rows only
relations:
- decomposes: epic:bert-tiny-step-tenth
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T21:54:45Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-10-28T21:54:45Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-10-29T07:18:18Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# MLM head and loss over the masked rows only

## Outcome

A BERT-tiny step on Vulkan computes the vocabulary projection, its bias, the cross-entropy loss
and its gradient only for the rows whose loss weight is non-zero (the masked positions, about 15 %
of 4,096), gathered at a fixed capacity per batch. Loss, gradients and parameters after a step are
bit-identical to the full-row computation, because zero-weight rows add exact zeros to every
ascending sum. Held-out evaluation downloads or tallies only the gathered rows.

## Context

After lane A the logits path is CrossEntropy 35.7 ms, CrossEntropyGrad 3.9 ms, three
4096 × 128 × 8192 matmuls 1.9 ms, AddRows/SumRows about 0.6 ms per step; the vocabulary projection
is 25.8 of the step's 39.4 GFLOP. Weights are 0/1 (`dataset/mlm.rs:784`), mask rate 0.15
(`spec/domains/data.yaml:167`). Explorer notes: scratch `explore/e4.md`.

## Scope

The IR or language form the gather takes (a `gather_rows`/compaction op, or a lowering the
compiler applies where a weighted cross-entropy follows a projection), CPU oracle, SPIR-V emit,
schedule, the data binding's capacity, `examples/bert_tiny.mspec` if the language form needs it,
DESIGN.md § 5 and § 9.

## Acceptance

- 3 Vulkan steps of `examples/bert_tiny.mspec` give the recorded bits of
  `bert_tiny_three_vulkan_steps_give_the_recorded_bits` (unchanged).
- A batch whose masked-row count exceeds the capacity is refused with a named error, never
  truncated.
- 200 steps, profiled, before and after, with the logits-path kernels' ms per step.
