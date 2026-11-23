---
format: aep.planning-md/3
id: story:resident-native-training
kind: story
status: implemented
title: Train an in-memory ModelSpec program across application batches
summary: The native training session owns parameters and AdamW moments across caller-provided batches and exposes them at checkpoints.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/execution.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan
- confidence: cited
  path: website/docs/concepts/training.md
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-11-23T01:42:04Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-11-23T01:42:04Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-11-23T02:01:14Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"test_result":1}}}
---
# Train an in-memory program across application batches

## Outcome

A Rust application supplies a new named minibatch and an explicit learning rate for each AdamW update. A Vulkan session retains learned parameters and both moments until the application requests a checkpoint.

## Acceptance

- Given the same initial state and three named float/integer batches, CPU and Vulkan return matching pre-update losses and final parameters and moments.
- A snapshot taken after the first update resumes the same subsequent trajectory, including optimizer bias correction.
- A normal Vulkan update submits one pre-recorded command buffer and downloads four bytes for its scalar loss when its controls are supplied in the batch.
- An out-of-range external ID fails before mutation; reusing the session succeeds on a valid batch.
- A resumed state with missing slots is rejected.
