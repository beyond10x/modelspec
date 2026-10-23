---
format: aep.planning-md/3
id: story:vulkan-train-compiles-once
kind: story
status: draft
title: Vulkan train compiles once
scope:
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-core/src/compile.rs
revision: 2
---
# Vulkan train compiles once

## Why

Review of 0.5.0 (finding P5), still open at main 496e98e. `train --backend vulkan` calls
`compile_training` (`crates/modelspec-cli/src/train.rs:372`) and then `compile_artifact` (`:393`),
which compiles the training again (`crates/modelspec-core/src/compile.rs:74`).

Outcome: one compile per `train` run.

## Acceptance

- **compile-artifact-reuses**: `compile_artifact` has a form that takes the `CompiledTraining` already
  built, and `train` uses it.
- **profile-counts-one-compile**: `--profile`'s `compile_us` covers one compile, and a test counts one
  call of `compile_training` per Vulkan run.
- **same-artifact**: the artifact hash is unchanged for the examples.
- `task ci` exits 0.

## Scope

- `crates/modelspec-cli/src/train.rs`
- `crates/modelspec-core/src/compile.rs`

## Out of scope

- Pipeline creation time, which story:vulkan-pipeline-cache holds.
