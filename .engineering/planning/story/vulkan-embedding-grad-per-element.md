---
format: aep.planning-md/3
id: story:vulkan-embedding-grad-per-element
kind: story
status: draft
title: EmbeddingGrad with one invocation per (row, column)
summary: Same per-element order, bit-identical; about -9 ms per step
relations:
- decomposes: epic:bert-tiny-step-tenth
- serves: vision:modelspec
revision: 1
---
# EmbeddingGrad with one invocation per (row, column)

## Outcome

EmbeddingGrad on Vulkan runs one invocation per (embedding row, column) and keeps the
per-element order of additions, so the gradient is bit-identical.

Measured by the diagnosis at 3b9869f (profiled): 11.0 ms per step at batch 32 × 128.

## Acceptance

1. A device test: the embedding gradients of 3 bert_tiny steps on Vulkan are bit-identical
   before and after.
2. The existing embedding tests pass unchanged.
3. A 200-step `--profile` run reports EmbeddingGrad at most 2 ms per step.
