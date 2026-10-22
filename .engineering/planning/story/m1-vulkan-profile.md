---
format: aep.planning-md/3
id: story:m1-vulkan-profile
kind: story
status: draft
title: 'Inspectable GPU runs: per-dispatch timestamps and run counters with train --profile'
summary: Timestamp queries per dispatch read back at run end with no extra host wait, run counters on stderr, and a guide covering IR, kernels, plan and timings
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:m1-vulkan-train
- depends_on: story:m1-compile-artifact
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/train_vulkan.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_profile.rs
- confidence: inferred
  path: website/docs/guides/inspect-a-run.md
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/sidebars.ts
revision: 7
---
# Inspectable GPU runs: per-dispatch timestamps and run counters with `train --profile`

## Context

M1 checkbox 9 of DESIGN.md § 9: "Built kernels, IR, dispatch graph, and performance timings are
inspectable". After the other M1 stories three of the four are: `modelspec ir --stage
tensor|training|kernel` prints the IR and both plans (`story:m1-kernel-plan`), and `modelspec
compile` writes every built kernel as a `.spv` file with its manifest (`story:m1-compile-artifact`).
Timings are missing. § 9 lists the performance metrics: steady-state examples per second, step
latency, per-kernel GPU time, number of dispatches, host and device transferred bytes, and compile
time, with warm-up and pipeline creation kept apart from steady state. § 5.2 has M2 tune "with GPU
timestamp queries"; `DeviceProfile.timestamp_queries` records whether the device has them.

`epic:pytorch-10x-mnist` needs these numbers per run to find where the 51.9 us step goes, so
profiling must not change what it measures: it adds no host wait. DESIGN.md § 6, as
`story:m1-vulkan-train` leaves it, takes per-kernel timing from timestamp queries on sampled
steps (from § 6.3 item 5 of revision 0.9); in M1 `--profile` samples every step, which the counts
below assume.

## What it adds

- `modelspec train ... --backend vulkan --profile`: a timestamp before and after each dispatch of
  every step, written into a query pool and read back with the results at the end of the run
  (no extra wait). Stderr gets one line per dispatch site, that is per Dispatch step of a plan:
  `profile dispatch <plan> <position> <kernel_module_id> <entry> calls <n> total_us <t> mean_us
  <m>`, with `<plan>` `step` or `forward` and `<position>` the index of the step in that plan. Two
  Dispatch steps that name one content-hashed module are two lines. Then one line each for
  dispatches, submits, host waits, bytes uploaded, bytes downloaded, pipeline creation time, compile
  time and steady-state time per step after a 20-step warm-up. Stdout is unchanged. On a device
  without timestamp queries the dispatch lines are replaced by `profile kernel timings unavailable:
  no timestamp queries` and the counters are still printed.
- `website/docs/guides/inspect-a-run.md`: the four inspection paths of checkbox 9 on
  `examples/mnist_mlp.mspec`, with the commands, written by hand.

## Domain relations

None: the profile is printed, not stored; a typed run report is later work of
`epic:pytorch-10x-mnist`.

## Scope

- `crates/modelspec-runtime/src/vulkan/` (query pool and profile), `crates/modelspec-runtime/tests/vulkan_profile.rs`
  (new).
- `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/tests/train_vulkan.rs`, both from
  `story:m1-vulkan-train`, which lands first. `story:m1-vulkan-checkpoint-infer` edits `train.rs`
  and the Vulkan runner loop after this story.
- `website/docs/guides/inspect-a-run.md` (new), `website/sidebars.ts` if the guide list is not
  generated, and through `task docs-generate` `website/docs/reference/cli.md`.
- `DESIGN.md` § 8 (`--profile` and the line format). The revision line, the § 8 verb table and
  `CHANGELOG.md` are append-only surfaces: this story adds its entry, and the wave coordinator
  assigns the revision number at merge.
- `CHANGELOG.md`.

## Acceptance

A profiled GPU run prints a time for every dispatch site and the run counters, with the same host
waits and the same stdout as an unprofiled run, and the guide shows all four inspection paths.

1. Operator run on the RTX 3090 (the operator starts it): `cargo test --release -p
   modelspec-runtime --features vulkan --test vulkan_profile -- --ignored` passes. For
   `examples/mnist_mlp.mspec`, `train --backend vulkan --profile` prints exactly one `profile
   dispatch` line per Dispatch step of the step plan and of the forward plan (the count is per
   dispatch site, not per module, so two sites that share a module are two lines); every `step`
   line has `calls` 1,800 and every `forward` line `calls` 1; and the sum of `total_us` over all
   lines is at most the measured wall time of the run.
2. In that run, `RunStats` host waits with `--profile` equal those without (1), and stdout with and
   without `--profile` is byte-equal.
3. In that run the counter lines match `RunStats`: dispatches equal 1,800 times the step-plan
   dispatch count plus the forward-plan count, and the byte counts equal those of
   `story:m1-vulkan-train` acceptance 2. The output is attached to the verification evidence.
4. `cargo test -p modelspec-cli --test train_vulkan`, in `task ci`: `train --backend cpu --profile`
   exits 2 with a usage error naming `--profile` as Vulkan only.
5. `task website` exits 0 with `website/docs/guides/inspect-a-run.md`, and that page holds four
   fenced shell blocks with the command lines `modelspec ir --stage kernel`,
   `modelspec compile ... --output`, `spirv-dis` on one kernel and
   `modelspec train examples/mnist_mlp.mspec --backend vulkan --profile`, the last followed by the
   output of the operator run of item 1; `grep -c` of each of the four command lines on the page
   is at least 1.
6. `task ci` exits 0, `task docs-check` included, with `--profile` on the CLI page.

## Out of scope

- Sampling a subset of steps: M2 picks the sample when the per-step timestamps cost more than they
  tell.
- A typed, stored benchmark report and the 10x gate (`epic:pytorch-10x-mnist`).
- Register and occupancy figures from vendor profilers (DESIGN.md § 5.2 M2).
- Peak VRAM accounting beyond the bytes the runtime allocates.
