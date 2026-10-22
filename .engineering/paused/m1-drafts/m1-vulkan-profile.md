# Inspectable GPU runs: per-kernel timestamps and run counters with `train --profile`

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
profiling must not change what it measures: it adds no host wait.

## What it adds

- `modelspec train ... --backend vulkan --profile`: a timestamp before and after each dispatch of
  every step, written into a query pool and read back with the results at the end of the run
  (no extra wait). Stderr gets one line per kernel (`profile kernel <id> <entry> calls <n>
  total_us <t> mean_us <m>`), then one line each for dispatches, submits, host waits, bytes
  uploaded, bytes downloaded, pipeline creation time, compile time and steady-state time per step
  after a 20-step warm-up. Stdout is unchanged. On a device without timestamp queries the kernel
  lines are replaced by `profile kernel timings unavailable: no timestamp queries` and the counters
  are still printed.
- `website/docs/guides/inspect-a-run.md`: the four inspection paths of checkbox 9 on
  `examples/mnist_mlp.mspec`, with the commands, written by hand.

## Domain relations

None: the profile is printed, not stored; a typed run report is later work of
`epic:pytorch-10x-mnist`.

## Scope

- `crates/modelspec-runtime/src/vulkan/` (query pool and profile), `crates/modelspec-runtime/tests/vulkan_profile.rs`
  (new).
- `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/tests/train_vulkan.rs`.
- `website/docs/guides/inspect-a-run.md` (new), `website/sidebars.ts` if the guide list is not
  generated, `website/docs/reference/` through `task docs-generate`.
- `DESIGN.md` § 8 (`--profile`), revision line bumped. `CHANGELOG.md`.

## Acceptance

A profiled GPU run prints a time for every kernel and the run counters, with the same host waits
and the same stdout as an unprofiled run, and the guide shows all four inspection paths.

1. Operator run on the RTX 3090 (the operator starts it): `cargo test --release -p
   modelspec-runtime --features vulkan --test vulkan_profile -- --ignored` passes. For
   `examples/mnist_mlp.mspec`, `train --backend vulkan --profile` prints one `profile kernel` line
   per kernel module of the artifact, each with `calls` 1,800, or 1,801 for a module the forward
   plan also dispatches, and
   the sum of `total_us` over the kernels is at most the measured wall time of the run.
2. In that run, `RunStats` host waits with `--profile` equal those without (1), and stdout with and
   without `--profile` is byte-equal.
3. In that run the counter lines match `RunStats`: dispatches equal 1,800 times the step-plan
   dispatch count plus the forward-plan count, and the byte counts equal those of
   `story:m1-vulkan-train` acceptance 2. The output is attached to the verification evidence.
4. `cargo test -p modelspec-cli --test train_vulkan`, in `task ci`: `train --backend cpu --profile`
   exits 2 with a usage error naming `--profile` as Vulkan only.
5. `task website` exits 0 with `website/docs/guides/inspect-a-run.md`, which runs `modelspec ir
   --stage kernel`, `modelspec compile ... --output`, `spirv-dis` on one kernel and `train
   --backend vulkan --profile` on `examples/mnist_mlp.mspec`.
6. `task ci` exits 0.

## Out of scope

- A typed, stored benchmark report and the 10x gate (`epic:pytorch-10x-mnist`).
- Register and occupancy figures from vendor profilers (DESIGN.md § 5.2 M2).
- Peak VRAM accounting beyond the bytes the runtime allocates.
