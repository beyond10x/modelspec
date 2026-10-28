---
format: aep.planning-md/3
id: epic:bert-tiny-beats-torch-compile
kind: epic
status: draft
title: Full BERT-tiny faster than torch.compile per step and to target
summary: Median us per step and seconds to the held-out baseline below torch.compile's, through whole-step fusion and memory planning, f32.
relations:
- decomposes: initiative:bert-tiny-beats-pytorch
- depends_on: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
revision: 1
---
## Outcome

`examples/bert_tiny.mspec` trains on the RTX 3090 through Vulkan at batch 32 with a median µs per
step below `torch.compile`'s, and reaches the held-out unigram baseline (6.926537926202018 nats)
in fewer training seconds, both measured the same day with the unit 23 harness, at f32 with TF32
off.

## Context

- Step F of `initiative:bert-tiny-beats-pytorch`; starts after `epic:bert-tiny-parity-with-eager`
  is implemented.
- `torch.compile` fuses elementwise chains per step but still launches every step from the host
  and keeps attention unfused at this size. ModelSpec knows the whole training step, forward,
  backward and optimizer, before it runs; this epic turns that into fewer, larger kernels and less
  memory traffic.

## Stories (drafts, confirmed or dropped by the profile of step E)

1. Fused layer norm with its residual add and dropout, forward and backward.
2. Fused attention that never writes the full score matrix to memory, forward and backward.
3. Fused GELU feed-forward epilogues on the matmul kernels.
4. A memory plan for the whole step: buffer liveness and reuse sized to the step, not per op.
5. Many steps per submission with no host wait between them, held to the CPU run's losses.
6. A closing story: timed runs, report, Benchmarks page, pitch deck.

## Out of scope

Reduced precision as the headline (a cooperative-matrix f16 row may be added beside it); a model
larger than BERT-tiny; CUDA or any vendor API.
