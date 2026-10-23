---
format: aep.planning-md/3
id: story:vulkan-loss-slots-written-by-kernel
kind: story
status: draft
title: The loss kernel writes the step's loss slot
scope:
- confidence: inferred
  path: crates/modelspec-core/src/schedule.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
revision: 3
---
# The loss kernel writes the step's loss slot

## Why

Review of 0.7.1 (finding 6), at main 496e98e. After each step's dispatches the runner records a
compute-to-transfer barrier, a 4-byte copy of the loss into the step's slot and a transfer-to-compute
barrier (`crates/modelspec-runtime/src/vulkan/runner.rs:697-699`). Two barriers per step serialise
the recording for one float.

Outcome: the loss kernel writes the loss straight into step `k`'s slot, and the per-step transfer
disappears.

## Acceptance

- **no-transfer-barrier**: the recording of a multi-step submission holds no per-step transfer
  barrier and no per-step copy (host test over the recorded actions).
- **same-loss-curve**: the loss curve of the MNIST example is identical to before.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`

## Out of scope

- The loss sum's workgroup count, which story:fast-math-mode holds.
