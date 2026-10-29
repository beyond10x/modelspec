---
format: aep.planning-md/3
id: story:vulkan-stepwise-multi-step-submit
kind: story
status: active
title: Stepwise Vulkan training records several steps per submission
summary: A stepwise run waits once per K steps with the bits of the one-step run.
relations:
- decomposes: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
scope:
- confidence: inferred
  path: crates/modelspec-cli/src/train.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan/stepwise.rs
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T03:20:13Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T03:20:13Z", actor: "human:timo", revision: 3}
---
# Stepwise Vulkan training records several steps per submission

## Context

Profile 2026-10-29, perf/bert-step-tenth 9e448ada, `train examples/bert_tiny.mspec` 200 steps, packed-128, `--backend vulkan --profile`: steady 15.0 ms per step, 12.3 ms of it in kernels, 593 dispatches, 418 barriers and one submit and host wait per step. PyTorch eager 9.9 ms, torch.compile 4.9 ms (verification report of story:bert-tiny-vs-pytorch-cuda, training clock).

Kernel time is 12.3 of the 15.0 ms; the rest is the host side of 593 dispatches and the submit and host wait after every step of a stepwise run (dropout, AdamW, packed data, held-out evaluation).

## Outcome

A stepwise run records K steps (dropout masks, batch selection and AdamW step counts for each) into one submission and waits once per K steps, with held-out evaluations and checkpoints at their steps; the losses and every printed line are the bits of the one-step-per-submit run. K is chosen by the runner from device memory and the evaluation and checkpoint schedule.

## Acceptance

- a probe first: the steady µs per step with K = 8 on the profile above;
- the bit tests of the BERT step, the dropout, checkpoint, resume and non-finite device tests unchanged; a new device test that a run with K > 1 prints the lines of K = 1, red first;
- interleaved A/B, 3 pairs, ratio in the commit.
