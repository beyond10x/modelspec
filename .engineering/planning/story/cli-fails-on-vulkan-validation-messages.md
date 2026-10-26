---
format: aep.planning-md/3
id: story:cli-fails-on-vulkan-validation-messages
kind: story
status: draft
title: The CLI fails a device run on a validation-layer message
summary: Under MODELSPEC_VK_VALIDATION=1 the CLI prints every validation message the layer reports and exits 1, so a CLI device test can fail on a hazard; found by review-result adversary-eval-download-pass-1 F1.
relations:
- decomposes: epic:m1-vulkan-execution
- serves: vision:modelspec
revision: 1
---
# The CLI fails a device run on a validation-layer message

## Context

Adversary pass 1 of `story:vulkan-eval-download-per-batch` (review-result
`adversary-eval-download-pass-1`, F1): with `MODELSPEC_VK_VALIDATION=1` and synchronization
validation on, a planted write-after-write hazard in the evaluation copies still gave exit 0 and
no stderr from the CLI, on the unit's head and on its base. `crates/modelspec-runtime/src/vulkan/mod.rs:299`
installs the debug messenger, but the CLI never reads what it collects, so every CLI device test
that sets the variable, `vulkan_eval_per_batch` included, cannot fail on a hazard; only the
in-process runtime tests can.

## What this story decides

- When `MODELSPEC_VK_VALIDATION=1` is set, `train`, `infer` and `compile --backend vulkan` print
  every validation message the layer reports on stderr, prefixed `validation:`, and exit 1 after
  the run with a one-line count, so a CLI device test under the variable fails on a hazard.
- Without the variable nothing changes.

## Scope

- `crates/modelspec-runtime/src/vulkan/mod.rs` (the messenger's collected messages exposed),
  `crates/modelspec-cli/src/train.rs`, `infer.rs`, `compile.rs`, their tests, AGENTS.md's GPU
  section.

## Acceptance

A CLI device run with a planted hazard under `MODELSPEC_VK_VALIDATION=1` exits 1 and prints the
layer's message; the same run without the variable is unchanged.

## Out of scope

- Changing which validation features are enabled.
