---
format: aep.planning-md/3
id: story:vulkan-step-counters
kind: story
status: implemented
title: Vulkan run counters for barriers, device allocations and peak device bytes
relations:
- serves: vision:modelspec
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/adversary_step_counters.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/memory.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/profile.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/sync.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_profile_throughput.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_step_counters.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/vulkan_profile.rs
- confidence: cited
  path: website/docs/guides/inspect-a-run.md
- confidence: cited
  path: website/docs/reference/cli.md
revision: 10
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T08:09:09Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "proposed", to: "active", at: "2026-10-24T08:09:09Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "active", to: "implemented", at: "2026-10-24T09:37:29Z", actor: "human:timo", revision: 10, decided_on: {"recorded":{"test_result":1,"review_outcome":3,"verification":1}}}
---
# Vulkan run counters for barriers, device allocations and peak device bytes

## Why

Review of 0.7.1, its performance-suite line ("GPU dispatches/barriers/bytes/VRAM per step"), and
critic round 1 on the review-finding stories: story:fuse-elementwise-chains-and-gradient-sums reads
a barrier count per step, and story:vulkan-buffer-liveness-suballocation reads a device allocation
count and peak device bytes, and no item creates any of the three. `RunStats`
(`crates/modelspec-runtime/src/vulkan/runner.rs:74-96`) counts steps, host waits, submits,
dispatches, dispatches per step and bytes moved; its `--stats` lines (`runner.rs:107-119`) and the
`--profile` counter lines (`crates/modelspec-runtime/src/vulkan/profile.rs:257-261`) print only
those. DESIGN.md § 8 lists the `--stats` lines (DESIGN.md:495). Barriers are recorded in one place
(`crates/modelspec-runtime/src/vulkan/sync.rs:64-80`) and device memory is allocated in one place
(`crates/modelspec-runtime/src/vulkan/memory.rs:190`).

New `RunStats` fields break the tests that build it as an exhaustive literal
(`crates/modelspec-runtime/tests/vulkan_profile.rs:52`,
`crates/modelspec-runtime/tests/adversary_profile_throughput.rs:26`), and new `--profile` lines
break the byte-pinned counter lines (`vulkan_profile.rs:188-193`, `:212-217`) and the counter block
the guide quotes (`website/docs/guides/inspect-a-run.md:129-136`). The `--stats` and `--profile`
help (`crates/modelspec-cli/src/train.rs:191-199`) lists what they print and generates
`website/docs/reference/cli.md:147-148`.

Outcome: `train --backend vulkan --stats` prints `barriers per step`, `device allocations` and
`peak device bytes`, each counted on the run it describes, so the two stories above can read them
by name.

## Acceptance

- **barriers-per-step**: `--stats` prints `barriers per step: <n>`, the pipeline barriers one
  training step records; a test on a plan with a known barrier count reads that count.
- **device-allocations**: `--stats` prints `device allocations: <n>`, the `vkAllocateMemory` calls
  of the run; a test on a plan with a known buffer count reads that count.
- **peak-device-bytes**: `--stats` prints `peak device bytes: <n>`, the largest sum of live device
  allocation sizes during the run; a test on a plan with known buffer sizes reads that sum.
- **profile-lines**: `--profile` prints the same three as `profile barriers_per_step`,
  `profile device_allocations` and `profile peak_device_bytes` among its counter lines.
- **pins-and-docs-updated**: the `RunStats` literals and the pinned profile lines in
  `vulkan_profile.rs` and `adversary_profile_throughput.rs`, the guide's counter block, and the
  `--stats` and `--profile` help with the generated `website/docs/reference/cli.md` name the three
  counters, and their tests and drift checks pass.
- **design-lists-lines**: DESIGN.md § 8 lists the three lines with the other `--stats` and
  `--profile` counters.
- `task ci` exits 0.

## Scope

- `crates/modelspec-cli/src/train.rs`
- `crates/modelspec-cli/tests/adversary_step_counters.rs`
- `crates/modelspec-runtime/src/vulkan/memory.rs`
- `crates/modelspec-runtime/src/vulkan/profile.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-runtime/src/vulkan/sync.rs`
- `crates/modelspec-runtime/tests/adversary_profile_throughput.rs`
- `crates/modelspec-runtime/tests/adversary_step_counters.rs`
- `crates/modelspec-runtime/tests/vulkan_profile.rs`
- `DESIGN.md`
- `website/docs/guides/inspect-a-run.md`
- `website/docs/reference/cli.md`

## Out of scope

- Reducing any of the three counts; story:fuse-elementwise-chains-and-gradient-sums and
  story:vulkan-buffer-liveness-suballocation do that.
- Counters on the CPU backend.
