---
format: aep.planning-md/3
id: story:profile-begin-stamp-after-barrier
kind: story
status: draft
title: Profile begin stamp follows the barrier
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: website/docs/guides/inspect-a-run.md
revision: 6
---
# Profile begin stamp follows the barrier

## Why

Review of 0.5.0 (finding C10), still open at main 496e98e. `--profile` writes each dispatch's begin
timestamp at `TOP_OF_PIPE` (`crates/modelspec-runtime/src/vulkan/dispatch.rs:560-564`, recorded from
`crates/modelspec-runtime/src/vulkan/runner.rs:1045`). That stage does not wait for the preceding
barrier, so a kernel's measured time includes the wait for the kernel before it.

Outcome: a profiled kernel's time starts when its own work can start.

## Acceptance

- **begin-stamp-stage**: the begin timestamp is written at `COMPUTE_SHADER`, or after the barrier,
  and a host test of the recording asserts the stage.
- **guide-numbers-retaken**: the profile figures in `website/docs/guides/inspect-a-run.md` are
  retaken on the RTX 3090, and the page says which commit they came from.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/vulkan/dispatch.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `website/docs/guides/inspect-a-run.md`

## Out of scope

- Changing what `--profile` reports beyond the begin stamp.
