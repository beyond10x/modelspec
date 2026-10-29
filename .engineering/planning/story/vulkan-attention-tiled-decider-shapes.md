---
format: aep.planning-md/3
id: story:vulkan-attention-tiled-decider-shapes
kind: story
status: implemented
title: Decider attention on the tiled Vulkan kernels, same bits
summary: 'Attention forward and gradients for the decider shapes run workgroup-tiled instead of per_row: 251 of 440 ms per step today.'
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T09:24:39Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T09:24:39Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-29T19:23:55Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1}}}
---
## Outcome

The decider model's attention runs through workgroup-tiled Vulkan kernels, forward and all three
gradients, with the bits it has today.

## Evidence that motivates it

`modelspec train --profile` of `models/decider.mspec` (decider repository) on an RTX 3090, 20 steps,
65,536 tokens per step, at perf/bert-eager-parity 50ec6c2b: 440 ms of device time per step, of which
attention takes 251 ms (57 %). `Attention` (8 sites, 88.6 ms), `AttentionGradK` (4, 64.5 ms),
`AttentionGradQ` (4, 61.9 ms) and `AttentionGradV` (4, 34.6 ms) run as `per_row` kernels on
Q, K, V `[65536, 256]` with a `[256, 256]` mask, at 1.6 to 2.0 TFLOP/s. BERT-tiny's attention
(0.23.0) runs one workgroup per sequence, head and tile; the decider's shapes do not take that path.

## Acceptance

- The decider attention sites compile to the tiled kernels (the kernel plan names them), or the
  story records why a shape cannot and what it runs instead.
- Forward and the three gradients give the same bits as before on the decider shapes and on
  BERT-tiny (device tests against the CPU reference and against the previous kernels).
- Device time of the four attention kinds on the decider model, median of alternating A/B pairs at
  P0: at most a third of 251 ms per step.
- BERT-tiny step time is not worse (A/B, 200 steps).
