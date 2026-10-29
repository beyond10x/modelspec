---
format: aep.planning-md/3
id: story:vulkan-attention-softmax-stats-reused
kind: story
status: implemented
title: Attention backward reads the forward's softmax statistics instead of recomputing them
summary: AttentionGradK and GradV read each query's max, sum and row dot from a buffer the forward and GradQ write once; bit-identical; about -149 ms per step
relations:
- decomposes: epic:bert-tiny-step-tenth
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T20:18:23Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-28T20:18:24Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-29T07:18:16Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1}}}
---
# Attention backward reads the forward's softmax statistics instead of recomputing them

## Outcome

AttentionGradK and AttentionGradV on Vulkan no longer recompute, inside every key invocation,
each query's softmax max, sum and row dot. The forward attention kernel (max, sum) and the GradQ
kernel (row dot) write them once per (sequence, head, query) to a buffer the backward kernels
read. The arithmetic per value is the same sequence of instructions, so every result is
bit-identical to today's Vulkan run.

Measured by the diagnosis at 3b9869f (profiled): GradK 120.3 ms and GradV 60.3 ms per step at
batch 32 × 128; with the three values replaced by constants, 5.1 ms and 4.9 ms.

## Acceptance

1. A device test trains `examples/bert_tiny.mspec` for 3 steps on Vulkan before and after: every
   loss and every parameter after step 3 is bit-identical (the test compares against values
   recorded from the current kernels).
2. The existing attention gradient tests (CPU oracle comparisons, adversary suites) pass
   unchanged.
3. A 200-step `--profile` run of bert_tiny reports AttentionGradK plus AttentionGradV at most
   20 ms per step (from 180.6); the before and after figures and their commands are in the
   commit message.
4. The statistics buffer is part of the step plan's buffer accounting; the plan verifier refuses
   a backward kernel that reads it before the forward writes it.
