---
format: aep.planning-md/3
id: epic:bert-tiny-parity-with-eager
kind: epic
status: draft
title: Full BERT-tiny at parity with PyTorch eager per step
summary: Median us per step of bert_tiny.mspec at batch 32 on Vulkan at most PyTorch eager's, same day, f32.
relations:
- decomposes: initiative:bert-tiny-beats-pytorch
- depends_on: epic:transformer-block-within-3x
- serves: vision:modelspec
revision: 1
---
## Outcome

`examples/bert_tiny.mspec` trains on the RTX 3090 through Vulkan at batch 32 with a median µs per
step (training seconds over steps, `train --clock`) at most PyTorch eager's, measured the same day
on the same device and driver with the unit 23 harness (`opponents/bert_tiny_pretrain/`, three
timed runs per side), at f32 with TF32 off.

## Context

- Step E of `initiative:bert-tiny-beats-pytorch`; starts after `epic:transformer-block-within-3x`
  is implemented.
- Starting point: the timed runs of `story:bert-tiny-vs-pytorch-cuda` (release 0.22.0).
- The full model adds to the block what the block bench does not run: the embedding lookup and its
  scatter gradient, the masked-word head and `cross_entropy` over the 8,192-token vocabulary,
  dropout masks, AdamW over 1,486,720 parameters, and held-out evaluation (outside the clock).

## Stories, in order

1. **Profile full BERT-tiny**: `train --backend vulkan --profile` at batch 32 on a committed tree,
   every site counted, ranked by µs per step, in a `verification-report`, with the sites the block
   profile of step A did not cover named; the next stories drafted from its ranking.
2. One story per measured site or cause. Candidates to confirm or drop: whole-step recording into
   one command buffer per chunk of steps for training with dropout and AdamW; the embedding
   scatter gradient without atomics contention; the vocabulary-sized head matmul and
   cross-entropy fused; dropout masks drawn on the device; AdamW as one kernel over all
   parameters.
3. A closing story: the timed runs of the final tree, the report, the Benchmarks page.

## Out of scope

Reduced precision as the headline; a model larger than BERT-tiny; CUDA or any vendor API.
