---
format: aep.planning-md/3
id: story:standin-targets-run-under-validation
kind: story
status: draft
title: Stand-in and old-device test targets run under MODELSPEC_VK_VALIDATION=1
relations:
- serves: vision:modelspec
revision: 1
---
# Stand-in and old-device test targets panic under `MODELSPEC_VK_VALIDATION=1`

## Why

Adversary pass 1 on story:matmul-reads-transposed-operand
(review-result:adversary-transposed-pass-1, finding F4): with `MODELSPEC_VK_VALIDATION=1`, 10
cases in 4 stand-in or old-device test targets (`crates/modelspec-cli/tests/inspect_device_stand_in.rs`
among them) panic inside ash with "Unable to load create_debug_utils_messenger_ext" and then
"panic in a function that cannot unwind"; without the variable all 11 of their cases pass.
Pre-existing: the unit did not change those files. The whole-workspace device run with
validation on therefore cannot be used as a gate until these targets either skip validation for
a stand-in instance or load the debug-utils function only when the instance has it.

Outcome: `MODELSPEC_VK_VALIDATION=1 cargo test --release --workspace -- --include-ignored`
reports no panic from a stand-in or old-device target; a stand-in instance without
`VK_EXT_debug_utils` runs without the messenger and says so once on stderr.

## Acceptance

- **stand-in-runs-under-validation**: the 10 cases pass with `MODELSPEC_VK_VALIDATION=1` and
  without it; a test asserts the messenger is created only when the instance has the extension.
- **no-abort**: no "panic in a function that cannot unwind" in the run; the messenger callback
  is `extern "system"` and never unwinds.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/vulkan/instance.rs`
- `crates/modelspec-cli/tests/inspect_device_stand_in.rs`
