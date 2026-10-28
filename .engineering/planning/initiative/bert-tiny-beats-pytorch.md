---
format: aep.planning-md/3
id: initiative:bert-tiny-beats-pytorch
kind: initiative
status: draft
title: 'Full BERT-tiny faster per step than PyTorch: parity with eager, then below torch.compile'
summary: 'Steps E and F after the 3x goal: full BERT-tiny on Vulkan at f32, first at parity with PyTorch eager, then below torch.compile, measured with the unit 23 harness.'
relations:
- serves: vision:modelspec
- depends_on: initiative:transformer-within-3x-of-pytorch
revision: 1
---
## Outcome

ModelSpec trains full BERT-tiny (`examples/bert_tiny.mspec`, batch 32, sequences of 128, AdamW,
dropout) on the RTX 3090 through Vulkan at f32 faster per step than PyTorch, first at parity
with eager, then below `torch.compile`, every figure measured on a committed tree with the
unit 23 harness and published on the Benchmarks page.

## Context

- Operator, 2026-10-28: after the 3× goal, "we also need step 2 and 3": parity with PyTorch
  eager on full BERT-tiny, then beating `torch.compile`.
- Follows `initiative:transformer-within-3x-of-pytorch`; its step A (the 3× block goal) comes
  first, and the profile unit of A ranks the sites both steps here start from.
- Smoke figures before the timed runs of `story:bert-tiny-vs-pytorch-cuda` (not results): about
  216 ms per step for ModelSpec against about 8 ms eager and about 4.1 ms `torch.compile`, about
  173 against 4,700 and 9,100 GFLOP/s on the matmul-only count of 37,446,746,112 per step. The
  MLP sweep reaches 15,635 GFLOP/s at large sizes (release 0.20.0), so the large-matmul kernel is
  not the gap.

## The order

1. **E** `epic:bert-tiny-parity-with-eager`: full BERT-tiny median µs per step at most PyTorch
   eager's.
2. **F** `epic:bert-tiny-beats-torch-compile`: full BERT-tiny median µs per step below
   `torch.compile`'s, and time to the held-out baseline below it too.

## Rules

Those of `initiative:transformer-within-3x-of-pytorch` § Rules: one release after every landed
unit, an adversary pass on every unit, timed device runs the coordinator's (the GPU alone, clocks
never locked, three timed runs, the median), numerics within the DESIGN.md § 9 run tolerance of
the CPU reference. Headline figures are f32 with TF32 off on both sides; a reduced-precision
variant is its own row, never the headline.

## Acceptance

- E: a `verification-report` whose same-day rows for `bert-tiny-pretrain` give ModelSpec Vulkan
  median µs per step at most PyTorch eager's.
- F: a `verification-report` whose same-day rows give ModelSpec Vulkan median µs per step and
  seconds to target below `torch.compile`'s.
