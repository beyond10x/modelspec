---
format: aep.planning-md/3
id: story:vulkan-eval-metrics-on-device
kind: story
status: implemented
title: Held-out evaluation tallied on the device
summary: The stepwise Vulkan run takes the evaluation loss sum from the CrossEntropy kernel and the accuracy from an argmax-count kernel, two floats per batch, one host wait per evaluated step.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:vulkan-eval-download-per-batch
- decomposes: epic:bert-tiny-step-tenth
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T21:37:44Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-28T21:37:44Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-29T07:18:17Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1}}}
---
# Held-out evaluation tallied on the device

## Context

`story:vulkan-eval-download-per-batch` makes the stepwise Vulkan run evaluate the HeldOut split
one batch at a time, downloading every batch's logits (134 MB for `examples/bert_tiny.mspec`) to
tally the loss and accuracy on the host: 68 downloads and 68 host waits per evaluated step on the
packed WikiText-103 corpus. Correct, and the form DESIGN.md § 5.2 asks for first.

## What this story decides

- The loss of an evaluated batch comes from the `CrossEntropy` kernel's loss sum on the device,
  and its accuracy from a kernel that counts the rows whose argmax equals the label with the CPU
  tally's tie rule, so two floats come down per batch and the whole evaluation runs in one
  command buffer with one host wait.
- The `eval` loss and accuracy equal the host tally's as `f32` shortest round-trip text on the
  twin and on `examples/bert_tiny.mspec` over the packed corpus.

## Scope

- `crates/modelspec-runtime/src/spirv/emit/`, `schedule.rs`, `vulkan/stepwise.rs`, their tests,
  DESIGN.md § 5.2 and § 8.

## Acceptance

An evaluated step of `examples/bert_tiny.mspec` on the packed corpus downloads at most
`8 × batches` bytes and takes one host wait, and prints the same `eval` line as the host tally.

## Out of scope

- Any change to what is evaluated.
