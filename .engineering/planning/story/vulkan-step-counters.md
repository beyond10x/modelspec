---
format: aep.planning-md/3
id: story:vulkan-step-counters
kind: story
status: draft
title: Vulkan run counters for barriers, device allocations and peak device bytes
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/memory.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/profile.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/sync.rs
revision: 2
---
# Vulkan run counters for barriers, device allocations and peak device bytes

## Why

Critic round 1 on the review-finding stories: story:fuse-elementwise-chains-and-gradient-sums reads
a barrier count per step, and story:vulkan-buffer-liveness-suballocation reads a device allocation
count and peak device bytes, and no item creates any of the three. `RunStats`
(`crates/modelspec-runtime/src/vulkan/runner.rs:74-96`) counts steps, host waits, submits,
dispatches, dispatches per step and bytes moved; its `--stats` lines (`runner.rs:107-119`) and the
`--profile` counter lines (`crates/modelspec-runtime/src/vulkan/profile.rs:257-261`) print only
those. DESIGN.md § 8 lists the `--stats` lines (DESIGN.md:495). Barriers are recorded in one place
(`crates/modelspec-runtime/src/vulkan/sync.rs:64-80`) and device memory is allocated in one place
(`crates/modelspec-runtime/src/vulkan/memory.rs:190`).

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
- **design-lists-lines**: DESIGN.md § 8 lists the three lines with the other `--stats` and
  `--profile` counters.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-runtime/src/vulkan/profile.rs`
- `crates/modelspec-runtime/src/vulkan/sync.rs`
- `crates/modelspec-runtime/src/vulkan/memory.rs`
- `DESIGN.md` (§ 8)

## Out of scope

- Reducing any of the three counts; story:fuse-elementwise-chains-and-gradient-sums and
  story:vulkan-buffer-liveness-suballocation do that.
- Counters on the CPU backend.
