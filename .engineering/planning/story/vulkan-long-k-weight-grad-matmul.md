---
format: aep.planning-md/3
id: story:vulkan-long-k-weight-grad-matmul
kind: story
status: active
title: Long-reduction weight-gradient matmuls near forward throughput, CPU summation order kept
summary: '[N,R]x[R,M] with R = 65,536 tokens runs at 4-5 TFLOP/s against 19-22 for forward matmuls: 87 of 440 ms per decider step.'
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T09:24:39Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T09:24:39Z", actor: "human:timo", revision: 3}
---
## Outcome

A Vulkan matmul whose reduction length is far larger than its output, the weight gradient
`[N, R] x [R, M]` with `R` the token count, runs near the throughput of the forward matmuls while
every output keeps the CPU's summation order and bits.

## Evidence that motivates it

`modelspec train --profile` of the decider model (decider repository, `models/decider.mspec`) on an
RTX 3090 at perf/bert-eager-parity 50ec6c2b, 65,536 tokens per step: `MatMul [256,65536]x[65536,256]`
(16 sites, 35.0 ms per step, 3.9 TFLOP/s), `[256,65536]x[65536,1024]` (4, 26.3 ms, 5.2 TFLOP/s) and
`[1024,65536]x[65536,256]` (4, 26.2 ms, 5.3 TFLOP/s): 87 ms, 20 % of the step. The forward matmuls of
the same step reach 18.6 to 22.1 TFLOP/s. A `[256, 256]` output has only 65,536 elements, so few
workgroups are resident while each walks 65,536 products.

## Acceptance

- Every output element of these kinds is the CPU reference's bits (same summation order); no
  split-K that reorders the sum, unless DESIGN.md records a numerics change the CPU reference makes
  too, which this story does not do.
- Device time of the three kinds on the decider model, median of alternating A/B pairs at P0: at
  most half of 87 ms per step.
- BERT-tiny and transformer-block steps are not worse (A/B), and their device bit tests pass.
