# `modelspec train --backend vulkan`: a whole training run on the GPU with one host wait

## Context

M1 checkboxes 6 and 7 of DESIGN.md § 9: "Forward, backward, parameter update execute on GPU
without CUDA API or toolkit usage" and "GPU/CPU results are numerically consistent within
specified tolerances". M0 refuses `train --backend vulkan` with E-TARGET-001
(`crates/modelspec-cli/src/train.rs`, test
`crates/modelspec-cli/tests/train_linear.rs::the_vulkan_backend_is_refused_in_m0`).

`epic:pytorch-10x-mnist` sets the later gate at 51.9 us per step on the RTX 3090, where PyTorch
eager CUDA took 519 us. M1 stays correctness first with naive kernels, but its runtime must not
close that path: many steps go into one command buffer with no host round trip per operator or
per step, the data and the weights stay on the device, and results come back at the end or at a
requested checkpoint. DESIGN.md § 6 still allows a block "at train-step completion"; this story
narrows it to run completion.

## What it adds

- `modelspec train <file> --backend vulkan [--device <index>]`: compiles the artifact in memory as
  `story:m1-compile-artifact` does, initializes the parameters on the host from the seed as the CPU
  run does, uploads parameters and the synthetic batch once, records the step plan of
  `story:m1-kernel-plan` for every step and after each step a copy of the loss into slot `k` of a
  device loss buffer, then the forward plan for the final loss. It submits command buffers of up to
  256 steps each without waiting between them, waits once on a fence, and reads back the loss
  buffer and the parameters. Stdout has the same lines as the CPU run.
- `RunStats` from the runtime: host waits, submits, dispatches, bytes uploaded and downloaded, and
  steady-state time per step; a test reads them, and stderr prints them with `--stats`.
- DESIGN.md § 6: the initial implementation blocks only at the end of a run and at a requested
  checkpoint, never per operator or per step.
- E-TARGET-001 stays for a binary built without the `vulkan` feature; the status page entries
  "Headless Vulkan runtime", "SPIR-V kernels for the M0 operators and SGD" and "End-to-end f32
  training on an RTX 3090" become shipped, each on the test that holds it.

## Domain relations

- A run reads its artifact (TrainingRun `artifact`, `spec/domains/training.yaml:132`) and its
  device (`device`, line 137). Inferable from ess/1. This story writes no TrainingRun record.

## Scope

- `crates/modelspec-runtime/src/vulkan/` (a runner module), `crates/modelspec-runtime/src/lib.rs`,
  `crates/modelspec-runtime/src/train/runner.rs`.
- `crates/modelspec-runtime/tests/vulkan_train.rs` (new), `crates/modelspec-runtime/tests/no_cuda.rs`
  (new).
- `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/tests/train_linear.rs`,
  `crates/modelspec-cli/tests/train_vulkan.rs` (new).
- `crates/modelspec-docs/src/status.rs`, `crates/modelspec-docs/src/diagnostics.rs`,
  `website/docs/roadmap.md`, `website/docs/getting-started.md`, `website/docs/reference/` and
  `website/data/` through `task docs-generate`.
- `DESIGN.md` § 6 and § 8, revision line bumped. `AGENTS.md` (State, Commands), `README.md` (Status line),
  `CHANGELOG.md`.

## Acceptance

`examples/mnist_mlp.mspec` and `examples/linear_regression.mspec` train on the RTX 3090 with one
host wait per run, the data and weights crossing the bus once each way, and final parameters within
the gate-5 tolerance of the CPU run.

1. Operator run on the RTX 3090 (the operator starts it): `MODELSPEC_VK_VALIDATION=1 cargo test
   --release -p modelspec-runtime --features vulkan --test vulkan_train -- --ignored` passes; for
   both examples every parameter after the last step satisfies `||g - c||_2 <= 1e-4 * ||c||_2`
   against the CPU run of the same file and seed, each of the `steps + 1` losses passes
   `GPU_ATOL + GPU_RTOL * |c|`, and the validation layer reports 0 messages.
2. The same run on `examples/mnist_mlp.mspec` (1,800 steps): `RunStats` reads 1 host wait, at most
   8 submits (`ceil(1800 / 256)`, the final forward plan in the last), 520,864 bytes uploaded (parameters 203,264, `x` 313,600,
   `y` 4,000) and 210,468 bytes downloaded (parameters 203,264, 1,801 losses 7,204). A planted
   wait after each step fails this item.
3. The same run prints the steady-state time per step and the dispatch count per step; both go
   into the verification evidence. No threshold: speed is `epic:pytorch-10x-mnist`.
4. Operator run: `modelspec train examples/mnist_mlp.mspec --backend vulkan` exits 0 and prints the
   same number of `init`, `step` and `param` lines, in the same order, as `--backend cpu`.
5. `cargo test -p modelspec-runtime --features vulkan --test no_cuda`, in `task ci`: the dependency
   tree of `modelspec-runtime` and `modelspec-cli` with all features names no crate whose name
   contains `cuda`, `cudarc`, `cust` or `nvrtc`, and the dynamic section of the built `modelspec`
   binary names no `libcuda`, `libcudart` or `libnvrtc`.
6. `cargo test -p modelspec-cli --test train_linear`: built without the `vulkan` feature,
   `train --backend vulkan` still exits 1 with E-TARGET-001; with it, on a host without a loader,
   it exits 1 with E-TARGET-002.
7. `task ci` exits 0, `task docs-check` included, with the three status entries shipped.

## Out of scope

- A gated speed target, fused kernels, tiled matmul and a whole-run schedule
  (`epic:pytorch-10x-mnist`).
- On-device minibatch gather and real MNIST data (`train --data`,
  `story:bert-train-data-binding`).
- Checkpoints and inference: `story:m1-checkpoint-cpu`, `story:m1-vulkan-checkpoint-infer`.
- Timestamp profiling: `story:m1-vulkan-profile`.
