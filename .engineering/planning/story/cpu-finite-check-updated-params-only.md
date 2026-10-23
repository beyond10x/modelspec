---
format: aep.planning-md/3
id: story:cpu-finite-check-updated-params-only
kind: story
status: draft
title: The CPU finite check scans only what a step changed
relations:
- informed_by: story:train-refuses-non-finite-loss
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
revision: 2
---
# The CPU finite check scans only what a step changed

## Why

Review of 0.7.1 (finding 12, in part), at main 496e98e. After each CPU step the runner checks the
loss and scans every value for a non-finite element
(`crates/modelspec-runtime/src/train/runner.rs:237`, scan at `:112`). The Vulkan backend scans only at
read-back.

Outcome: the per-step CPU check reads only the loss and the parameters the step updated, with the
same failure semantics as story:train-refuses-non-finite-loss.

## Acceptance

- **same-failure**: every test of story:train-refuses-non-finite-loss passes unchanged, with the same
  step, code and paths reported.
- **scan-limited**: a test shows the per-step scan visits only the loss and the updated parameters.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/train/runner.rs`

## Out of scope

- The Vulkan check (story:train-vulkan-refuses-non-finite-loss).
