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
revision: 3
---
# Attention kernels: one workgroup per (sequence, head), keys and values in workgroup memory

## Outcome

The four attention kernels (forward, GradQ, GradK, GradV) on Vulkan run one workgroup of 128
invocations per (sequence, head, tile of rows) instead of one invocation per (row, head), which
filled about 3 warps per SM on the RTX 3090 with 8,192 invocations. Each output keeps its order
of additions, so results are bit-identical; where a reordering cannot be avoided, the story
stops and records it as a decision before changing any bit.

The sequence length 128 of bert_tiny and the lengths 256 and 32 of the decider model, each with
a key mask, are covered and tested.

## Acceptance

1. A device test: 3 bert_tiny steps on Vulkan are bit-identical before and after.
2. The existing attention tests pass unchanged.
3. A 200-step `--profile` run reports the four attention kernels together below their figure
   after story:vulkan-attention-softmax-stats-reused, with before and after in the commit
   message.

## Design taken

- Grid: workgroup `(sequence · heads + head) · tiles + tile`, 128 invocations
  (`schedule::ATTENTION_WORKGROUP`), over the `T` rows of one tile of one sequence's head.
  `T = schedule::attention_tile(S)` is the largest power of two of at most 32 and `S` whose
  staged values fit the 16 KiB of workgroup memory every device has: 32 for `S = 32`, 8 for
  128, 4 for 256, 2 for 512. `Attention` and `AttentionGradQ` tile query rows, `AttentionGradK`
  and `AttentionGradV` key rows. Dispatch counts and bindings are unchanged.
- Weights in workgroup memory: the workgroup computes the `[T, S]` weights of its rows once,
  between barriers: scores, then probabilities, and for `AttentionGradQ` the `dP` beside them
  and `dS` in place; for the key gradients one invocation per query computes `P` or `dS` for
  every key row of the tile. The softmax maximum and sum and the row dot of a query row are
  computed by one invocation of the row, in ascending key order, or read from the shared
  statistics of story:vulkan-attention-softmax-stats-reused.
- Held heads and unrolled dot products: the invocation of a key (or query) holds that row's
  head of `d <= 128` values in registers and computes its dot product with each row of the tile
  in straight-line code, every load before the first product, the sum fused in ascending
  column order from `-0.0`.
- Output in registers, written once: each of the `128 / T` invocations of a row owns the
  columns `lane + u · lanes` and keeps one sum per column in its own variable, loading each
  weight once for all of them, then writes each column once.
- Mask-free summing: `AttentionGradK` and `AttentionGradV` sum without a mask check, up to 32
  values loaded ahead of their products. `Attention` and `AttentionGradQ` do the same when the
  mask keeps every key of the sequence, a flag the workgroup sets before it sums (a validated
  guard inside the one kernel, so the kernel identity is unchanged); otherwise they skip
  masked keys as before.
- Not done: a merged `AttentionGradK` + `AttentionGradV` dispatch.
