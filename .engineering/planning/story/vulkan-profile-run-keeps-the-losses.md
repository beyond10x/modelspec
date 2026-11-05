---
format: aep.planning-md/3
id: story:vulkan-profile-run-keeps-the-losses
kind: story
status: draft
title: train --profile prints the losses of the run without it
summary: 'Finding 2026-11-05: on Vulkan a --profile run of BERT-tiny prints losses that differ from the run without it and between profiled runs; non-profiled runs are bit-stable. Cause unknown.'
relations:
- serves: vision:modelspec
revision: 3
---
# `train --profile` prints losses that differ from a run without it

## Context

Found 2026-11-05 while measuring the BERT-tiny step (epic:bert-tiny-beats-torch-compile), on the
RTX 3090, on the tree before and after the kernel stories of that day (not checked on a release
binary): `train examples/bert_tiny.mspec --backend vulkan --data
<packed-128>` at 200 steps prints the same losses and parameters on every run without `--profile`
(step 200 loss 8.04707 on three runs), and with `--profile` it prints losses that differ from
those and from one profiled run to the next in the third or fourth digit. The cause is not known.
A profiled run records one timestamp query around every dispatch; whether those queries reorder
or split a recording is the first thing to check.

## Outcome

A run with `--profile` prints the losses and parameters of the same run without it, bit for bit,
or DESIGN.md § 8 states why it cannot and by how much it may differ.

## Acceptance

- A test trains the hidden-32 BERT twin for 20 steps on the device with and without `--profile`
  and compares every loss line and the final parameters bit for bit (`--ignored`, it needs a
  Vulkan device).
- The cause is named in the story's result with the observation that shows it.
