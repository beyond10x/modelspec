---
format: aep.planning-md/3
id: story:vulkan-validation-without-debug-utils
kind: story
status: draft
title: Validation mode refuses a loader without debug utils instead of aborting
summary: Under MODELSPEC_VK_VALIDATION=1 a loader without VK_EXT_debug_utils gets a refusal, not a process abort.
relations:
- decomposes: epic:m1-vulkan-execution
- serves: vision:modelspec
revision: 1
---
# Validation mode refuses a loader without debug utils instead of aborting

## Context

With `MODELSPEC_VK_VALIDATION=1`, `Vulkan::load` (`crates/modelspec-runtime/src/vulkan/mod.rs`)
enables `VK_EXT_debug_utils` and the validation layer unconditionally. On the stand-in loaders the
device tests use (Vulkan 1.0/1.1 stand-ins), the extension is missing and the process aborts with
a panic in a function that cannot unwind, leaving a core file. 10 stand-in tests and the
`adversary2_create_device_old` target abort this way on 0.22.0 and on perf/bert-step-tenth; they
pass without the variable. Found by the gate triage on 2026-10-29.

## Outcome

Under validation, a loader without the layer or `VK_EXT_debug_utils` gives an `E-TARGET-003`
refusal naming what is missing (or runs without validation and says so), never an abort.

## Acceptance

- The stand-in targets (`adversary2_inspect_pre_1_2`, `adversary_inspect_old_device`,
  `inspect_device_stand_in`, `adversary2_create_device_old`) exit without SIGABRT under
  `MODELSPEC_VK_VALIDATION=1`, red first.
