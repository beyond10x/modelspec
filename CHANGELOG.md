# Changelog

All notable changes to ModelSpec are recorded here, in the format of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). ModelSpec releases from source at
bare-version tags such as `0.1.0`; nothing is on a registry. Each release on GitHub carries the
`modelspec` CLI binary for Linux x86_64 and its SHA-256 checksum. Changes collect under
`[Unreleased]` until the next release; the detail of each change is in its story and commit
history.

## [Unreleased]

### Added

- `benchmarks/runs.yaml` holds every benchmark run measured so far (51: the MNIST 784-64-10 runs of ModelSpec CPU and Vulkan and of the PyTorch CPU and CUDA opponents, and the 36 points of the MLP sweep), typed by the ESS domain `modelspec.benchmark`; `modelspec-docs` checks it and generates the page Benchmarks (`website/docs/benchmarks.md`) with a chart per workload of µs per step over time, ModelSpec beside the opponents.

### Changed

- A CPU training run prepares its step once (`train::optimizer::PreparedStep`, `cpu::interpreter::Prepared`): the step program is verified once per run instead of once per step (5 verifier calls over 5 steps become 1), the `effects_in_order`, writes and in-place-transpose tables are built once, each intermediate is freed after its last consumer (peak live values on the MNIST step 23 of 25 nodes become 10), storage is reused across steps (170 allocations in the second step become 4, the outputs' shape vectors) and outputs are moved out instead of copied; `modelspec train --backend cpu` output on `examples/mnist_mlp.mspec` and `linear_regression.mspec` has the same sha256 as before, and a value listed twice among the step's outputs, as two parameters sharing one gradient are, keeps the loss at position 0 and both gradients; the MNIST CPU bench measured 0.383 ms per step before and 0.384 ms after at the median of 101 interleaved rounds, inside that host's noise, the AVX2 matmul kernel being 75–79 % of a step (`crates/modelspec-runtime/tests/prepared_step.rs`, `prepared_run.rs`) (story:cpu-prepared-training-step)
- The parser places a line-and-column anchor at offset 0, at every lexeme start and at the end of input in one pass, and `span_at` resolves a diagnostic's position from the nearest anchor by binary search instead of counting characters from the line start, so a span's cost no longer grows with its column: on a 20,000-`let` program written on one line the lookup scans 309,257 characters where it scanned 9,087,562,188, the same count as the one-`let`-per-line program, and `modelspec check` on that file takes 0.977 of the per-line file's time in release (`crates/modelspec-core/tests/span_lookup.rs`; `span_lookup_scan_count` is a `doc(hidden)` test hook) (story:parser-span-lookup-not-linear-in-column)
- `modelspec train --backend vulkan --stats` and `--profile` print three per-step counters from a per-thread ledger in `modelspec_runtime::vulkan`: pipeline barriers recorded in the step, device allocations, and the peak device bytes live at once over the run (a buffer freed inside the span counts once, one freed after it is excluded); on `examples/mnist_mlp.mspec` a step records 7 barriers and 12 allocations, `--stats` and `--profile` print the same values in one run, and checkpoints and `--resume` leave the per-step figures unchanged; DESIGN.md § 8 (revision 0.30) lists the lines in the order the code prints them (story:vulkan-step-counters)
- A dense-data training step refills its `x` and `y` batch tensors in place through `HostTensor::refill`, which keeps the shape when it is equal and clears and refills the data, instead of allocating a new key, shape and data per step: a runner step on dense data makes 104.125 allocations where it made 111.125, the same as a synthetic step, and the per-step losses of 1,800 MNIST steps are bit-identical (`crates/modelspec-runtime/tests/dense_batch_reuse.rs`) (story:dense-step-reuses-batch-buffers)
- The CPU `matmul` writes its output without filling it with zeros first: the portable kernel appends rows to the output `Vec`, and the AVX2 f32 and f64 kernels write into reserved spare capacity and set the length to the rows they stored, so no memset of `m·n` elements precedes the kernel; a `k` of 0 still zero-fills, because those zeros are the result, and an element type without vector code in the module runs its own `matmul_avx2` hook on a zero-filled output as before, counted by the `doc(hidden)` test instrument `output_zero_fills` (`crates/modelspec-runtime/tests/matmul_output_fill.rs`, `matmul_output_zero_fills.rs`) (story:cpu-matmul-output-skips-zero-fill)

## [0.7.1] - 2026-10-23

Throughput you can measure, and a Vulkan descriptor pool sized right: the bench `mlp_sweep` trains 784 -> H ReLU -> 10 at 12 sizes on one Vulkan device beside a PyTorch CUDA opponent, every Dispatch step of a plan has a FLOP and a byte count that `ir --stage kernel` and `train --backend vulkan --profile` print, and the descriptor pool counts every view a dispatch reads and writes. On an RTX 3090, `train --backend vulkan --profile` on `examples/mnist_mlp.mspec` reached 447.8 GFLOP/s and 56.6 GB/s over 45.925 µs per profiled step, with 0 uncounted sites; the sweep measured 36 of 36 points, ModelSpec on Vulkan faster than PyTorch eager CUDA at every batch-100 point (28.1 µs against 537.5 µs at hidden 64) and PyTorch eager up to 4.26 times faster at batch 8192 (7290.5 µs against 31067.1 µs at hidden 4096). No new timed MNIST step time was measured for this release; the latest is 0.6.0's 27.14 µs per step, and ADR-007's accuracy and final-weight conditions were not measured.

### Added

- The bench `mlp_sweep` (`cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep`) trains 784 -> H ReLU -> 10 at hidden 64, 256, 1024, 4096 and batch 100, 1000, 8192 on one Vulkan device, each source formatted and compiled in Rust, and prints one JSON line per point: the matmul-only FLOP per step `3196·B·H`, the µs per step of 3 timed runs of at least 5 s after a 20-step warm-up, their median and GFLOP/s, and the version, git commit and dirty state it was built from, or the refusal of a point that does not compile or run, the teacher-student batch made once per point and handed to every run through the new `RunOptions::data`; `--crossover <file>...` reads its lines and those of the PyTorch CUDA opponent `opponents/mlp_sweep/mlp_sweep.py` (eager and `torch.compile`, TF32 off) and names per side and batch the first width more than 10 % slower than the previous, and `tests/mlp_sweep.rs` checks that all 12 sources compile for the RTX 3090 profile and that their step plans' matmuls count `3196·B·H` (story:mlp-size-sweep-vs-pytorch-cuda)
- Every Dispatch step of an execution plan has a FLOP and a byte count by the counting rule of DESIGN.md § 3.1 (`ExecutionPlan::dispatch_count`, `DispatchCount`, `Kernel::flop`): `2·m·k·n` for a matmul, the arithmetic per output element for an elementwise kernel and an SGD update, one per input element for a reduction or a loss sum, none for a transpose or a broadcast, and the bytes of the views the step reads and writes; a kernel without a rule is `uncounted`, never 0. The counts are computed when the plan is read, so no artifact field, hash or module id changes. `modelspec ir --stage kernel` prints `flop <n> bytes <n>` or `uncounted` on each dispatch line, and `modelspec train --backend vulkan --profile` adds `flop`, `bytes`, `gflop_per_s` and `gb_per_s` to each `profile dispatch` line and ends with `profile step_total`, the step's counted FLOP and bytes over the steady time per step and `uncounted_sites`, the step sites it leaves out (story:profile-achieved-throughput)

### Fixed

- `modelspec train --backend vulkan` and `infer --backend vulkan` size the descriptor pool of `modelspec_runtime::vulkan::runner` at every view each dispatch reads and writes, where it counted one write per dispatch: `examples/mnist_mlp.mspec` binds 39 storage-buffer descriptors from a pool of 39 instead of 38, which a driver that accounts the pool exactly could refuse with `VK_ERROR_OUT_OF_POOL_MEMORY` since 0.4.0; `DescriptorArena::bind`, `Recording::dispatch` and `Recording::dispatch_kernel` admit each descriptor set into the room their pool was created from and refuse one past it with `ExecError::DescriptorPoolFull` before the pool is asked, on every driver (story:vulkan-descriptor-pool-counts-writes)

## [0.7.0] - 2026-10-23

The observatory UI, and Vulkan runs and artifacts that refuse what they cannot trust: `--ui <dir>` on `train --serve` and `observe` serves a Vue app that draws the loss and every metric per step, a CPU run's events carry per-parameter norms and histograms of values and gradients, `train --backend vulkan` stops a diverged run with the CPU run's `E-TRAIN-001`, `artifact::read` refuses a plan past 4,096 steps, 65,536 dependencies or 65,536 views with `E-ARTIFACT-003` and verifies the rest in bounded time, and a matmul's kernel module id covers its `MatMulTile`. No new device benchmark was measured for this release; the latest measurement is 0.6.0's 27.14 µs per step on an RTX 3090.

### Added

- The observatory UI in `ui/` (Vue 3, TypeScript, Vite, pnpm; `task ui` builds `ui/dist`) draws the loss and every `MetricSample` per step from the event stream, and `--ui <dir>` on `modelspec train --serve` and `modelspec observe` serves it at `/` beside `/events` (`modelspec_observatory::Ui`; `spec/domains/observatory.yaml`, `UiDirectory`), refusing a directory over 4096 files or 64 MiB without reading more than 64 MiB of it; a page left open after the server is stopped mid-run connects again every second and shows the run of whatever server then answers at that address; its TypeScript types are generated from the Rust `RunEvent` types by `modelspec-ui-types` and drift-checked in `task ci` (story:observatory-ui-live-loss-chart)
- `modelspec train --backend cpu` fills `StepCompleted.parameters` at the first step, every `--statistics-every N` steps (default 100) and the final step: per parameter the L2 norm and a 32-bin histogram of its values and of its gradient, with NaN and ±inf counted beside the bins (`Histogram.nan_count`, `positive_infinity_count`, `negative_infinity_count`, new required fields), the final step and a parameter the loss does not read having no `Gradient` entry; `sgd_step` returns the step's gradients, and the UI draws a value-norm and a gradient-norm chart per parameter, a metric named `<path> value norm` or `<path> gradient norm` getting ` (metric)` as `loss` does, and the histograms of the latest or a picked sampled step; `--backend vulkan` fills the first and the final step with `Value` entries only, from the values already on the host, sends every other list empty and refuses `--statistics-every` as a usage error (story:observatory-parameter-statistics)

### Fixed

- `modelspec train --backend vulkan` checks every loss and the parameters it reads back, after the last step and before each `--checkpoint-every` checkpoint, and stops a diverged run with the CPU run's `E-TRAIN-001` diagnostic, stdout and `--serve` events instead of printing NaN parameters and exiting 0, with no host wait of its own; a parameter that goes non-finite between read-backs while its step's loss is finite is named at the first later step that shows it, so `sgd(lr = 1e38)` names the loss of step 1 where the CPU names the update of step 0, and `--stats` and `--profile` print the counters of the steps the device ran after the diagnostic (DESIGN.md § 8) (story:train-vulkan-refuses-non-finite-loss)
- `modelspec_core::artifact::read` refuses a plan of more than 4,096 steps (`MAX_PLAN_STEPS`), 65,536 dependencies (`MAX_PLAN_DEPENDENCIES`) or 65,536 views (`MAX_PLAN_VIEWS`) with `E-ARTIFACT-003` before its other invariant checks, and the plan verifier keeps each pass's state and the step orderings as bit sets instead of searching every earlier step, with the same result on every plan; a re-sealed artifact with a 300,000-step dependency chain is refused in 1.0 s to 1.2 s and at most 64 MiB in a debug build, where it ran past 10 s before, and a 4,096-step plan with 16 dependencies per step verifies in 0.09 s to 0.15 s instead of more than 10 s (DESIGN.md § 7.1, § 8.1) (story:plan-verifier-bounded-on-untrusted-artifact)
- `modelspec_core::artifact::read` counts the steps, dependencies and views of each plan as it parses `artifact.json` and refuses a plan past a limit there with `E-ARTIFACT-003`, without holding it and before the module files and hashes are checked, so an unsealed artifact with such a plan gets `E-ARTIFACT-003` instead of `E-ARTIFACT-002`; a sealed 64 MiB manifest of 33.5 million dependencies is refused in 2.4 s to 4.5 s and at most 71 MiB in a debug build, where it passed 1 GiB before; `artifact_hash` is computed as the manifest is serialized, without a copy of it; a manifest with a repeated key is refused with `E-ARTIFACT-001`; and the plan verifier names an unsynced view by looking the earlier step's buffers up in sets, so a pair of Copy steps at the view limit verifies in 0.02 s to 0.05 s instead of 3.4 s (DESIGN.md § 7.1, § 8.1) (story:plan-verifier-bounded-on-untrusted-artifact)
- A matmul kernel's `Kernel::signature`, and so its module id, ends with its `MatMulTile` (`tile workgroup_size … cols … rows … depth … ahead … buffers …`), so two matmuls that differ only in tile depth or ahead no longer print the same id in `modelspec ir --stage kernel` and `train --profile`; every matmul id changes, the SPIR-V and the bits do not (story:kernel-signature-covers-matmul-tile)

## [0.6.0] - 2026-10-23

Checkpoints, inference and artifacts on Vulkan: `train --backend vulkan` writes and resumes checkpoints, `infer --backend vulkan` runs a checkpoint on the device, `compile` writes a sealed artifact directory, and `train --record` with `observe` records and replays a run. The MNIST benchmark example measured 27.14 µs per step on an RTX 3090 (median of 5 timed runs of 144,000 steps), down from 27.282 µs in 0.5.0. ADR-007's accuracy and final-weight conditions were not measured for this release.

### Added

- `modelspec train --backend vulkan` takes `--checkpoint`, `--checkpoint-every` and `--resume`: each requested checkpoint ends a segment of the run with one host wait and a parameter read-back, so 1800 MNIST steps with `--checkpoint-every 600` wait 3 times and write 2 extra files, and a run resumed at step 900 passes against the uninterrupted one within the GPU tolerance; `modelspec infer --backend vulkan [--device <index>]` runs a checkpoint's forward program and loss on the device with one host wait, refusing an unusable checkpoint with the CPU reader's `E-CKPT` code before the loader opens (DESIGN.md § 6.2, § 8) (story:m1-vulkan-checkpoint-infer)
- `modelspec compile <file> --backend cpu|vulkan --output <dir> [--device-profile <json>]` writes an artifact directory, `artifact.json` and for Vulkan one `kernels/<kernel_module_id>.spv` per kernel, sealed by a blake3 `artifact_hash`, byte-equal across two compiles, replacing the `artifact.json` and `kernels/` of an earlier artifact in that directory; `modelspec_core::artifact::read` reads one back and refuses an altered one with `E-ARTIFACT-001` (format version), `E-ARTIFACT-004` (module file missing, not a regular file or too long, or a stray file in `kernels/`), `E-ARTIFACT-002` (hash mismatch) or `E-ARTIFACT-003` (broken invariant or plan, a buffer over 4294967295 bytes, a module that is not SPIR-V), in that order (DESIGN.md § 7.1, § 8) (story:m1-compile-artifact)
- `modelspec train --record <path>` writes the run's events to a recorded run file, one JSON `RunEvent` per line in `sequence` order (`modelspec_observatory::Recorder`), and `modelspec observe <path> [--serve <addr>]` replays it as the `text/event-stream` of `--serve`, the same frames and `id:` lines, refusing a truncated file, a `sequence` gap or two `training_run_id` values, an order `RunEventBody` forbids (a first event other than `RunStarted`, a second `RunStarted`, an event after `RunFinished` or `RunFailed`, a missing end) on one line with exit 1 before serving (`spec/domains/observatory.yaml`, `RecordedRunFile`) (story:observatory-run-recording-replay)

### Changed

- A two-row `MatMulTile` reads tiles of up to 16 values of `k` 2 tiles ahead (`modelspec_core::ir::kernel::MATMUL_TWO_ROW_TILE`) instead of 32 read 6 ahead, with the same bits; the weight-gradient matmul of the MNIST example falls from 4.91 to 4.58 µs and the MNIST step from 27.50 to 27.14 µs on an RTX 3090 (medians of 3 profiled and 5 timed runs of 144,000 steps) (DESIGN.md § 5.2) (story:m1-f-weight-grad-matmul-tiled)

### Fixed

- `modelspec train --backend cpu` stops at the first step whose loss or updated parameter is NaN or ±inf and exits 1 with `E-TRAIN-001` naming the step and the loss or parameter, instead of printing a diverged run and exiting 0; stdout ends with that step's `step <k> loss <v>` line and has no `param` lines, and `--serve` ends with that step's event (its loss `null` when the loss is the non-finite value), then `RunFailed` (DESIGN.md § 8) (story:train-refuses-non-finite-loss)
- A checkpoint is written to `<path>.tmp`, synced and renamed over `<path>`, so a write that fails part-way (a full disk, a file-size limit, a kill) keeps the checkpoint it would replace, including the `--resume` source when `--checkpoint` names the same file.

## [0.5.0] - 2026-10-23

Faster Vulkan training: the `MatMul` kernel stages its operand tiles in workgroup memory, and the MNIST benchmark example measured 27.282 µs per step on an RTX 3090 (median of 5 runs of 144,000 steps, mean P0 clock 1957 to 1965 MHz), down from 40.225 µs in 0.4.0. ADR-007's accuracy and final-weight conditions were not measured for this release.

### Changed

- The Vulkan `MatMul` kernel stages tiles of its operands in workgroup memory (`modelspec_core::ir::kernel::MatMulTile`, 128 invocations per block of 8 rows by 16 columns, or 16 rows at two per invocation when 8 need at least 256 workgroups) with the same bits as before, its tiles of `k` in a loop of 4-tile passes so a module's size does not grow with `k`; the forward matmul of the MNIST example falls from 18.5 to 8.3 µs per step and the MNIST step from 40.2 to 27.3 µs on an RTX 3090 (medians of 3 profiled and 5 timed runs of 144,000 steps), and a tile larger than the device's workgroup memory is refused with `EmitError::SharedMemory` (DESIGN.md § 5.2) (story:m1-f-forward-matmul-tiled)

## [0.4.0] - 2026-10-22

Training on a Vulkan GPU: `train --backend vulkan` runs a whole training run on one device with one host wait. The MNIST benchmark example measured 40.225 µs per step on an RTX 3090 (median of 5 runs of 144,000 steps, mean P0 clock 1935 to 1951 MHz). ADR-007's accuracy and final-weight conditions were not measured for this release.

### Added

- `examples/mnist_mlp.mspec`: the MNIST benchmark shapes (784 → 64 ReLU → 10, no bias, SGD lr 0.1, batch 100, 1,800 steps) on M0 synthetic data, checked by `task check-examples`, gradient-checked at sampled coordinates and with a training-IR golden file (story:m1-mlp-example)
- `modelspec inspect-device --backend vulkan [--device N] [--format text|json]`: the `modelspec_core::device::DeviceProfile` of each Vulkan device as the driver reports it and its negotiation verdict (Vulkan 1.2, numerics class), with codes E-TARGET-002 to E-TARGET-004 (story:m1-inspect-device)
- `modelspec ir --stage kernel` prints the Kernel IR of a training step: its buffer views, kernels with module ids, step plan and forward plan, built by `modelspec_core::schedule::schedule`, checked by `modelspec_core::ir::verify::verify_plan` (and `verify_recorded_back_to_back` for back-to-back recording) and wrapped by `compile::compile_artifact`; a tensor too large to dispatch or bind is refused with `E-TARGET-010` (story:m1-kernel-plan)
- `modelspec train --data <dir>`, `--init <dir>` and `--eval <dir>` train on dense `f32` tensor files with an optional `order.u32`, start parameters from files, and print `eval loss=… accuracy=…`; codes `E-DATA-007`, `E-DATA-008`, `E-DATA-009` (story:train-data-dense-tensors)
- `modelspec_runtime::spirv::emit` builds the SPIR-V module of each elementwise, broadcast and SGD kernel, checked by `spirv-val --target-env vulkan1.2` in `task ci`, and `modelspec_runtime::vulkan` gains device buffers, compute pipelines, one-command-buffer dispatch with a counted host wait, and `MODELSPEC_VK_VALIDATION=1` for the validation layer; GPU results must lie within `GPU_ATOL = 1e-5` plus `GPU_RTOL = 1e-4` times `|c|` (DESIGN.md § 9 gate 5) (story:m1-vulkan-elementwise)
- `modelspec_runtime::spirv::emit::emit_kernels` builds a `KernelModule` with entry point, SPIR-V and `KernelCacheKey` for every kernel of an artifact, now including the two-stage `ReduceSum`/`ReduceMean`, the `mse` loss sum, `Transpose` and `MatMul` in orders fixed by the shape alone, with `NoContraction` and, on a device that reports them, the RTE, denorm-preserve and signed-zero-preserve modes on loss sum and matmul; `Recording::dispatch_kernel` and `Recording::reduce` record them (DESIGN.md § 3.3, § 5.2, § 5.4, § 9 gate 5) (story:m1-vulkan-reduce-matmul)
- `modelspec train --steps`, `--checkpoint`, `--checkpoint-every` and `--resume` write and resume `.msckpt` checkpoints (`modelspec_runtime::train::checkpoint`, a 56-byte header, a JSON manifest and 64-byte aligned tensors under a payload hash), a resumed CPU run writing the same bytes as one that never stopped, and `modelspec infer --backend cpu --checkpoint` runs the forward program with a checkpoint's parameters; codes E-CKPT-001 to E-CKPT-004 (story:m1-checkpoint-cpu)
- `modelspec train --backend vulkan [--device N] [--stats]` trains on one Vulkan device with one host wait per run (`modelspec_runtime::vulkan::runner::train`: chained command buffers of `CHUNK_STEPS` = 256 steps, one upload, one read-back) and prints the lines of the CPU run within the gate 5 tolerance; `--stats` prints the `RunStats` counters on stderr, `train` reports E-TARGET-002 to E-TARGET-004, and only a build without the `vulkan` feature refuses the backend with E-TARGET-001 (story:m1-vulkan-train)
- `modelspec train --backend vulkan --profile` prints on stderr one `profile dispatch <plan> <position> <kernel_module_id> <entry> calls <n> total_us <t> mean_us <m>` line per dispatch site from timestamp queries read back with no extra host wait, then the run counters, pipeline creation and compile time and the steady-state µs per step after a 20-step warm-up (`modelspec_runtime::vulkan::runner::train_profiled`, `modelspec_runtime::vulkan::profile`); the guide `inspect-a-run` shows the IR, plan and timing paths (story:m1-vulkan-profile)

### Changed

- The `mse` loss sum kernel of `modelspec_runtime::spirv::emit` runs one workgroup of 64 invocations that stage the terms in workgroup memory, in chunks of `max_shared_memory_bytes / 4`, and invocation 0 adds them in the same row-major order, so the loss sum and its bits are unchanged; `KernelKind::LossSum.workgroup_size()` is now 64 and its kernel module id changes (story:m1-loss-sum-staged)
- `modelspec_core::schedule::schedule` fuses a `relu` of a `matmul` output into the matmul kernel (`Epilogue::Relu`, writing the pre-activation too when the `relu` gradient reads it), runs each SGD update `p - lr * g` as one `sgd_update` kernel that reads the gradient, and folds a `mul`, `add` or `sub` of two compile-time scalars into its reader's immediate (it stays a kernel when a reader takes no immediate), with the same bits: the MNIST step is 14 dispatches in 8 barrier groups instead of 18 in 11 (story:m1-fuse-elementwise)

## [0.3.0] - 2026-10-22

An AVX2 CPU `matmul`, 2.98× faster on the MNIST shape with the same bits.

### Changed

- CPU reference: `matmul` runs an AVX2 kernel with register blocks where the CPU has AVX2 (`modelspec_runtime::cpu::matmul::Kernel`), and `evaluate` reads inputs, parameters and a transposed left `matmul` operand in place, with the same bits as 0.1.1 (`cpu-matmul-avx2-blocking`)

## [0.2.0] - 2026-10-22

Live training events, input adjoints and a 15× faster CPU `matmul` with the same bits.

### Added

- `modelspec train --serve <addr>` streams the run as typed `RunEvent`s over Server-Sent Events at `http://<addr>/events` on a loopback address, in the new crate `modelspec-observatory` (story:observatory-live-event-stream)
- `modelspec_core::autodiff::differentiate_inputs` builds the gradient of a rank-0 value with respect to named inputs, and `output_element_target` builds the rank-0 target for one element of a model output (story:observatory-input-adjoint)

### Changed

- CPU reference: `matmul` runs in loop order `i`, `k`, `j` with the same bits as 0.1.1, and the `mnist_shape` benchmark (`cargo bench -p modelspec-runtime --bench mnist_shape`) prints images/s for the 784 → 64 → 10 shape (`cpu-matmul-row-order`)

## [0.1.1] - 2026-10-21

Documentation release: the website review's fixes and highlighted `modelspec` code blocks. The
compiler and runtime behave as in 0.1.0.

### Added

- Documentation site: `modelspec` code blocks are syntax-highlighted; the word lists are `website/data/modelspec.grammar.json`, generated from `syntax::lexer::KEYWORDS`, `resolve::OPERATORS` and the parser

### Changed

- Diagnostics reference: every code has its own anchor, such as `reference/diagnostics#e-type-012`
- CLI reference: an argument table without optional arguments drops the Required and Default columns, and each possible value of `--stage` and `--backend` has its own line
- Operators reference: the message of a refused call wraps between its parts
- `website/data/status.json` and the Status page: each shipped capability names the test or task that holds it
- Getting started: the install block uses `install -D` and says `~/.local/bin` must be on `PATH`; the binary links `libc`, `libm` and `libgcc_s`
- Roadmap: planned milestones carry the `planned` badge, and the exclusions list all of DESIGN.md § 1 and § 2.4
- Write a model: step 2 says where `forward` goes, and step 4 shows the `E-TYPE-012` output
- The language: `infer` is refused with `E-SYNTAX-002`; a top-level `target` is later, `target y:` in `train` is M0
- Pitch deck `docs/pitch/`: slides 13 and 15 show the 0.1.0 release, slide 8's loop arrows meet their boxes, tap, swipe and Escape work, and a portrait phone gets a notice

## [0.1.0] - 2026-10-21

The M0 semantic core (`epic:m0-semantic-core`, DESIGN.md revision 0.6). There is no GPU or Vulkan
backend yet: Vulkan execution is milestone M1, and `modelspec train --backend vulkan` is refused
with `E-TARGET-001`.

### Added

- A Cargo workspace of `modelspec-core`, `modelspec-runtime` and `modelspec-cli` (story:m0-workspace-foundation)
- `modelspec-core` has no Vulkan dependency under any feature or target; `task core-no-vulkan` holds it
- `modelspec_core::syntax::parse` and `syntax::print`: lexer and parser for the M0 subset (story:m0-parse-subset)
- Syntax errors `E-SYNTAX-001` to `E-SYNTAX-021`, each with line and column
- `modelspec_core::types::check_source`: name resolution, type and shape checking into `hir::Hir` (story:m0-check-types-shapes)
- Name errors `E-NAME-001` to `E-NAME-008` and type and shape errors `E-TYPE-001` to `E-TYPE-017`
- `modelspec_core::ir::tensor::TensorProgram` and `TensorProgramBuilder`: the tensor SSA IR (story:m0-tensor-ir-core)
- `TensorOp` registry: `Add`, `Sub`, `Mul`, `MatMul`, `Relu`, `Transpose`, `ReduceSum`, `ReduceMean`, `Mse`
- `ir::verify::verify` refuses a use before definition, an undeclared `ParamId` and a shape mismatch
- `ir::print::print`: a deterministic text form of the Tensor IR
- `modelspec_core::lower::lower`: Typed HIR to Tensor IR (story:m0-lower-tensor-ir)
- `modelspec_core::passes::dce::eliminate`: dead-code elimination over the Tensor IR
- `modelspec_core::autodiff::differentiate`: reverse-mode autodiff into a `TrainingProgram` (story:m0-reverse-mode-autodiff)
- `ir::training::attach_sgd` and `ir::verify::verify_step`: the verified SGD training step
- `modelspec_core::compile::compile_training`: source to a verified `CompiledTraining` in one call
- `modelspec_runtime::cpu::evaluate`: the CPU reference interpreter over Tensor IR, in `f32` and `f64` (story:m0-cpu-interpreter)
- `modelspec_runtime::train::runner::run`: seeded SGD training on the CPU interpreter (story:m0-cpu-sgd-training)
- `train::init`: Philox4x32-10 parameter initialization keyed by seed and canonical parameter path (DESIGN.md § 2.3)
- `train::data::synthetic`: M0 synthetic teacher data; data errors `E-DATA-001` to `E-DATA-003`
- `modelspec check <file>`: parse and check; diagnostics on stderr, exit 1
- `modelspec ir <file> --stage tensor|training`: print the verified Tensor IR or training step
- `modelspec train <file> --backend cpu`: train on the CPU reference, printing initial params, losses and final params
- CLI exit codes: 0 success, 1 diagnostics or runtime error, 2 usage error
- `examples/linear_regression.mspec` checks and trains; `task check-examples` runs `modelspec check` on it
- ESS contracts for the compiled artifact, checkpoint, device profile and training run in `spec/`

[Unreleased]: https://github.com/beyond10x/modelspec/compare/0.7.1...HEAD
[0.7.1]: https://github.com/beyond10x/modelspec/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/beyond10x/modelspec/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/beyond10x/modelspec/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/beyond10x/modelspec/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/beyond10x/modelspec/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/beyond10x/modelspec/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/beyond10x/modelspec/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/beyond10x/modelspec/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/beyond10x/modelspec/releases/tag/0.1.0
