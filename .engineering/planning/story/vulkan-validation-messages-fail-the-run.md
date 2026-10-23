---
format: aep.planning-md/3
id: story:vulkan-validation-messages-fail-the-run
kind: story
status: draft
title: The CLI fails a Vulkan run on a validation message
revision: 1
---
# The CLI fails a Vulkan run on a validation message

## Why

Validation-layer messages are queued by `keep_message` (`crates/modelspec-runtime/src/vulkan/mod.rs`)
and read only by runtime test files through `take_validation_messages`. Nothing in
`crates/modelspec-cli/src` or `vulkan/runner.rs` reads them, so a validation error during
`modelspec train --backend vulkan` is dropped, and the CLI device tests whose headers ask for
`MODELSPEC_VK_VALIDATION=1` (`checkpoint_vulkan.rs`, `train_vulkan.rs`,
`train_vulkan_non_finite.rs`) cannot fail on one. Source: code review of wave m1-h, finding 17
(PLAUSIBLE).

## Acceptance

- With `MODELSPEC_VK_VALIDATION=1`, the runner drains the queue after the host wait; any message
  fails the run with a named error that prints the messages.
- A host-only test pushes a message into the queue and reaches that error.
- `task ci` exits 0, and the ignored CLI Vulkan tests pass on the local GPU.
