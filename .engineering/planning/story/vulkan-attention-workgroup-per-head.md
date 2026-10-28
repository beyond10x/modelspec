---
format: aep.planning-md/3
id: story:vulkan-attention-workgroup-per-head
kind: story
status: draft
title: 'Attention kernels: one workgroup per (sequence, head), keys and values in workgroup memory'
summary: Fills the GPU instead of about 3 warps per SM; bit-identical or stop and decide; about -14 ms per step (estimated)
relations:
- decomposes: epic:bert-tiny-step-tenth
- depends_on: story:vulkan-attention-softmax-stats-reused
- serves: vision:modelspec
revision: 1
---
# Attention kernels: one workgroup per (sequence, head), keys and values in workgroup memory

## Outcome

The four attention kernels (forward, GradQ, GradK, GradV) on Vulkan run one workgroup per
(sequence, head), with that head's K and V rows in workgroup memory. Today's 8,192 invocations
fill about 3 warps per SM on the RTX 3090. Each output keeps its order of additions, so results
are bit-identical; where a reordering cannot be avoided, the story stops and records it as a
decision before changing any bit.

Estimated, not probed: about 14 ms per step after story:vulkan-attention-softmax-stats-reused.

## Acceptance

1. A device test: 3 bert_tiny steps on Vulkan are bit-identical before and after.
2. The existing attention tests pass unchanged.
3. A 200-step `--profile` run reports the four attention kernels together below their figure
   after story:vulkan-attention-softmax-stats-reused, with before and after in the commit
   message.
