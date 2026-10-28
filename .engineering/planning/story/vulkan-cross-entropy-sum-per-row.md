---
format: aep.planning-md/3
id: story:vulkan-cross-entropy-sum-per-row
kind: story
status: draft
title: 'CrossEntropy loss sum: per-row kernel, then one ordered sum'
summary: Replaces the single-workgroup serial loss sum; bit-identical; about -31 ms per step
relations:
- decomposes: epic:bert-tiny-step-tenth
- serves: vision:modelspec
revision: 1
---
# CrossEntropy loss sum: per-row kernel, then one ordered sum

## Outcome

The CrossEntropy loss sum on Vulkan stops running as one workgroup over 64 serial blocks. A
per-row kernel writes each row's weighted NLL; one invocation then adds the rows in ascending
row order, the order used today, so the loss is bit-identical.

Measured by the diagnosis at 3b9869f (profiled): 36.3 ms per step; with one block only, 0.6 ms.

## Acceptance

1. A device test: the loss of 3 bert_tiny steps on Vulkan is bit-identical before and after.
2. The existing cross-entropy tests (CPU oracle, adversary suites, NaN and infinity cases) pass
   unchanged.
3. A 200-step `--profile` run reports the loss-sum kernels at most 3 ms per step (from 36.3).
