---
format: aep.planning-md/3
id: story:train-vulkan-refuses-non-finite-loss
kind: story
status: draft
title: train --backend vulkan refuses a non-finite loss or parameter with E-TRAIN-001
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
revision: 1
---
# `train --backend vulkan` refuses a non-finite loss or parameter with E-TRAIN-001

## Context

`story:train-refuses-non-finite-loss` (wave m1-g) made `train --backend cpu` stop at the first step
whose loss, or a parameter after its update, is NaN or ±inf, and exit 1 with E-TRAIN-001. Its
out-of-scope line deferred the Vulkan backend "when its runner exists". The Vulkan runner,
`crates/modelspec-runtime/src/vulkan/runner.rs`, already exists and does no such check; DESIGN.md
§ 8 says "The Vulkan backend does not check yet". The wave m1-g adversary read, not run: a
diverging `--backend vulkan` run exits 0.

## What it changes

- The Vulkan runner checks the loss and the updated parameters of each step it reads back, with the
  CPU runner's order (loss first) and the same `RunError::NonFinite`, so the CLI output, the
  diagnostic and the `--serve` stream are those of the CPU backend.
- A check that needs a read-back the run does not already make must not change the timing of a run
  that stays finite beyond the ADR-007 noise; where it would, the check reads back at the steps the
  run already reads back and names the first such step.

## Scope

- `crates/modelspec-runtime/src/vulkan/runner.rs`, `crates/modelspec-cli/src/train.rs`, the Vulkan
  training tests, `DESIGN.md` § 8 and its revision line, `CHANGELOG.md`.

## Acceptance

1. `train --backend vulkan` on the `lr = 100.0` and `lr = 1e38` fixtures exits 1, stdout and stderr
   equal to the CPU backend's for the same file.
2. Under `--serve` the stream ends as the CPU backend's does.
3. A finite run's 1,800-step stdout sha256 over 5 runs equals the head's before the change.
4. The base and the head timed in one session by the ADR-007 method; the head's median is at most
   the base's plus the method's noise.
5. `task ci` exits 0.
