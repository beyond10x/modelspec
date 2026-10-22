---
format: aep.planning-md/3
id: story:m1-vulkan-train
kind: story
status: draft
title: 'modelspec train --backend vulkan: a whole training run on the GPU with one host wait'
summary: Data and weights uploaded once, many steps per command buffer, losses kept on the device, one wait and one read-back per run, final parameters within 1e-4 relative L2 of the CPU, no CUDA
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:m1-mlp-example
- depends_on: story:m1-vulkan-reduce-matmul
- depends_on: story:m1-checkpoint-cpu
- depends_on: story:observatory-live-event-stream
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: README.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/train_linear.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/train_vulkan.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-docs/src/example.rs
- confidence: cited
  path: crates/modelspec-docs/src/status.rs
- confidence: cited
  path: crates/modelspec-runtime/src/lib.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan
- confidence: inferred
  path: crates/modelspec-runtime/tests/no_cuda.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_train.rs
- confidence: cited
  path: website/data/modelspec.terminal.json
- confidence: cited
  path: website/data/status.json
- confidence: cited
  path: website/docs/examples/linear-regression.md
- confidence: cited
  path: website/docs/getting-started.md
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/docs/reference/diagnostics.md
- confidence: cited
  path: website/docs/roadmap.md
revision: 7
---
# `modelspec train --backend vulkan`: a whole training run on the GPU with one host wait

## Context

M1 checkboxes 6 and 7 of DESIGN.md § 9: "Forward, backward, parameter update execute on GPU
without CUDA API or toolkit usage" and "GPU/CPU results are numerically consistent within
specified tolerances". M0 refuses `train --backend vulkan` with E-TARGET-001
(`crates/modelspec-cli/src/train.rs`, test
`crates/modelspec-cli/tests/train_linear.rs::the_vulkan_backend_is_refused_in_m0`).

`epic:pytorch-10x-mnist` sets the later gate at 51.9 us per step on the RTX 3090, where PyTorch
eager CUDA took 519 us. M1 stays correctness first with naive kernels, but its runtime must not
close that path. This story is written against DESIGN.md revision 0.9 (the 10× patch). Its § 6
lets the initial implementation "block only at train-step completion" and adds "From M2 the runtime
does not block per step either (§ 6.3)"; its § 6.3 puts chunked recording with waits only at the
end, at a checkpoint and at a requested read-back (items 4 and 5) in M2. This story brings those two
items into M1 and leaves § 6.3 with the M2 increments only.

## What it adds

- `modelspec train <file> --backend vulkan [--device <index>]`: compiles the artifact in memory with
  `spirv::emit::emit_kernels` of `story:m1-vulkan-reduce-matmul` (no artifact on disk), initializes
  the parameters on the host from the seed as the CPU run does, uploads parameters and the synthetic
  batch once, records the step plan of `story:m1-kernel-plan` for every step and after each step a
  copy of the loss into slot `k` of a device loss buffer, then the forward plan for the final loss.
  It submits command buffers of up to 256 steps each without waiting between them, waits once on a
  fence, and reads back the loss buffer and the parameters. Stdout has the same lines as the CPU
  run.
- `RunStats` from the runtime: host waits, submits, dispatches, bytes uploaded and downloaded, the
  numerics class of DESIGN.md § 5.4 and steady-state time per step; a test reads them, and stderr
  prints them with `--stats`. In M1 a Vulkan run is in the tolerance class: only the loss-sum and
  matmul kernels declare the float-control modes (`story:m1-vulkan-reduce-matmul`).
- DESIGN.md § 6, on top of revision 0.9: the sentence on blocking at train-step completion and the
  sentence "From M2 the runtime does not block per step either (§ 6.3)" become: the runtime records
  steps in chunks, one command buffer per chunk, and the host waits only at the end of a run, at a
  requested checkpoint and at a requested read-back, never per operator or per step; parameters and
  the reported losses are read back at those points. § 6.3 items 4 and 5 move into § 6 as M1, and
  § 6.3 keeps only its M2 increments: the resident data set with on-device gather (items 1 and 2),
  two parameter versions (item 3), the fused schedule (item 6) and its budget (item 7).
- The numerics of a run follow `story:m1-vulkan-reduce-matmul`: every parameter after the last step
  passes the whole-run bound of § 9 gate 5 against the CPU run, and each loss passes the gate-5
  tolerance with the `r` that story defines for `mse`.
- The site stops recording a Vulkan run: `crates/modelspec-docs/src/example.rs` records
  `train --backend vulkan` on the linear-regression page and the landing terminal and expects exit
  1 with E-TARGET-001 (lines 77-79, 195 and the unit test at 270-275). After this story that output
  depends on the host GPU, and `task docs-check` would create a Vulkan instance. The run, its
  expected exit, the `E-TARGET-001:` tone and the three refusal assertions of the unit test go; the
  `--backend gpu` parse error and the missing-file case stay.
- E-TARGET-001 stays for a binary built without the `vulkan` feature; the shipped status entry
  "The Vulkan backend is refused in M0" becomes "A binary built without the vulkan feature refuses
  the Vulkan backend", held by the test of acceptance 6; the status page entries
  "Headless Vulkan runtime", "SPIR-V kernels for the M0 operators and SGD" and "End-to-end f32
  training on an RTX 3090" become shipped, each on the test that holds it.

## Domain relations

- A run references its artifact (relation `artifact` of `TrainingRun`, kind references, via
  `artifact_id`, `spec/domains/training.yaml`) and its device (relation `device`, via
  `device_profile_id`). Inferable from ess/1. This story writes no TrainingRun record.

## Scope

- `crates/modelspec-runtime/src/vulkan/` (a runner module), `crates/modelspec-runtime/src/lib.rs`,
  `crates/modelspec-runtime/src/train/runner.rs`.
- `crates/modelspec-runtime/tests/vulkan_train.rs` (new), `crates/modelspec-runtime/tests/no_cuda.rs`
  (new).
- `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/tests/train_linear.rs`,
  `crates/modelspec-cli/tests/train_vulkan.rs` (new).
- `train/runner.rs` and `cli/src/train.rs` are also edited by `story:observatory-live-event-stream`
  (wave observatory-1) and by `story:m1-checkpoint-cpu`; this story depends on both and is scoped
  against `main` after they merge.
- `crates/modelspec-docs/src/status.rs`, `crates/modelspec-docs/src/diagnostics.rs`,
  `crates/modelspec-docs/src/example.rs`, `website/docs/roadmap.md`,
  `website/docs/getting-started.md`, and through `task docs-generate`
  `website/docs/reference/cli.md`, `website/docs/reference/diagnostics.md`,
  `website/docs/examples/linear-regression.md`, `website/data/status.json` and
  `website/data/modelspec.terminal.json`.
- `DESIGN.md` § 6 and § 6.3 (on top of revision 0.9) and § 8. The revision line, the § 8 verb table
  and `CHANGELOG.md` are append-only surfaces: this story adds its entry, and the wave coordinator
  assigns the revision number at merge.
- `AGENTS.md` (State, Commands), `README.md` (Status line), `CHANGELOG.md`.
- No crate dependency is added: `no_cuda.rs` runs `cargo tree` and `readelf -d` as processes.

## Acceptance

`examples/mnist_mlp.mspec` and `examples/linear_regression.mspec` train on the RTX 3090 with one
host wait per run, the data and weights crossing the bus once each way, and final parameters within
the gate-5 tolerance of the CPU run.

1. Operator run on the RTX 3090 (the operator starts it): `MODELSPEC_VK_VALIDATION=1 cargo test
   --release -p modelspec-runtime --features vulkan --test vulkan_train -- --ignored` passes; for
   both examples every parameter after the last step passes the whole-run bound
   `||g - c||_2 <= 1e-4 * ||c||_2` of `story:m1-vulkan-reduce-matmul` against the CPU run of the
   same file and seed, each of the `steps + 1` losses passes `GPU_ATOL + GPU_RTOL * r` with the
   `mse` `r` of that story, `RunStats` names the class `tolerance`, and the validation layer reports
   0 messages.
2. The same run on `examples/mnist_mlp.mspec` (1,800 steps): `RunStats` reads 1 host wait, at most
   8 submits (`ceil(1800 / 256)`, the final forward plan in the last), 520,864 bytes uploaded
   (parameters 203,264, `x` 313,600, `y` 4,000) and 210,468 bytes downloaded (parameters 203,264,
   1,801 losses 7,204). A planted wait after each step fails this item.
3. The same run prints the steady-state time per step and the dispatch count per step; both go
   into the verification evidence. No threshold: speed is `epic:pytorch-10x-mnist`.
4. Operator run: `modelspec train examples/mnist_mlp.mspec --backend vulkan` exits 0 and prints the
   same number of `init`, `step` and `param` lines, in the same order, as `--backend cpu`.
5. `cargo test -p modelspec-runtime --features vulkan --test no_cuda`, in `task ci`: the dependency
   tree of `modelspec-runtime` and `modelspec-cli` with all features names no crate whose name
   contains `cuda`, `cudarc`, `cust` or `nvrtc`, and the dynamic section of the built `modelspec`
   binary names no `libcuda`, `libcudart` or `libnvrtc`.
6. `cargo test -p modelspec-cli --test train_linear`, in `task ci`: built without the `vulkan`
   feature, `train --backend vulkan` exits 1 with E-TARGET-001; built with it and with
   `MODELSPEC_VULKAN_LIBRARY` set to a path that does not exist, it exits 1 and its stderr starts
   with `E-TARGET-002: `. No test in `task ci` creates a Vulkan instance.
7. `task ci` exits 0, `task docs-check` included, with the three status entries shipped and no
   generated page or `website/data/modelspec.terminal.json` entry recording a `--backend vulkan`
   run; DESIGN.md § 6 states the M1 waits of this story and § 6.3 lists only the M2 increments.

## Out of scope

- A gated speed target, fused kernels, tiled matmul, two parameter versions and the resident data
  set with on-device gather (DESIGN.md § 6.3, `epic:pytorch-10x-mnist`).
- Real MNIST data (`train --data`, `story:bert-train-data-binding`).
- Loading a compiled artifact from disk: `story:m1-compile-artifact` writes one; `train` compiles
  in memory.
- Checkpoints and inference: `story:m1-checkpoint-cpu`, `story:m1-vulkan-checkpoint-infer`.
- Timestamp profiling: `story:m1-vulkan-profile`.
