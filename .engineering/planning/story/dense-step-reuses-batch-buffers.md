---
format: aep.planning-md/3
id: story:dense-step-reuses-batch-buffers
kind: story
status: draft
title: A dense step reuses its batch buffers
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
revision: 2
---
# A dense step reuses its batch buffers

## Why

Review of 0.5.0 (finding P3), still open at main 496e98e. `DenseFeed::step` builds a new map of new
tensors with new `String` keys for every step (`crates/modelspec-runtime/src/train/data.rs:207-239`),
called once per step from `crates/modelspec-runtime/src/train/runner.rs:206`.

Outcome: a dense training step fills buffers allocated once per run.

## Acceptance

- **no-per-step-allocation**: after the first step, a step allocates no `x`, `y` or key `String`
  (counted by a test allocator or by buffer identity across steps).
- **same-loss-bits**: the per-step loss of the MNIST example is bit-identical to before.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/train/data.rs`
- `crates/modelspec-runtime/src/train/runner.rs`

## Out of scope

- Copies inside the interpreter, which story:cpu-prepared-training-step holds.
