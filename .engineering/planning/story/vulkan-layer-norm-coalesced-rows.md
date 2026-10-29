---
format: aep.planning-md/3
id: story:vulkan-layer-norm-coalesced-rows
kind: story
status: implemented
title: LayerNorm and LayerNormGradX read rows coalesced, same bits
summary: 'The per_row kernels reach 86-125 GB/s against 491 GB/s for AddRows on [65536,256]: 40 of 440 ms per decider step.'
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T09:24:40Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T09:24:40Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-29T19:23:56Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1}}}
---
## Outcome

The Vulkan `LayerNorm` and `LayerNormGradX` `per_row` kernels read and write their rows coalesced,
keeping each row's summation order and bits.

## Evidence that motivates it

`modelspec train --profile` of the decider model (decider repository, `models/decider.mspec`) on an
RTX 3090 at perf/bert-eager-parity 50ec6c2b, 65,536 tokens per step: `LayerNorm` `[65536,256]`
(18 sites, 19.3 ms per step, 125 GB/s) and `LayerNormGradX` (9 sites, 21.0 ms, 86 GB/s), against
491 GB/s for `AddRows` on the same shape in the same step.

## Acceptance

- Same bits as before for LayerNorm, LayerNormGradX and LayerNormGradGamma's row moments, on the
  decider and BERT-tiny shapes (device tests against the CPU reference).
- Device time of the two kinds on the decider model, median of alternating A/B pairs at P0: at
  most half of 40.3 ms per step.
- BERT-tiny step time is not worse (A/B, 200 steps).
