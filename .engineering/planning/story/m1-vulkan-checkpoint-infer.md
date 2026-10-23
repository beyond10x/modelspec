---
format: aep.planning-md/3
id: story:m1-vulkan-checkpoint-infer
kind: story
status: implemented
title: Checkpoints of a GPU run, resume on the GPU, and modelspec infer --backend vulkan
summary: One host wait per requested checkpoint, GPU resume within tolerance of an uninterrupted run, GPU checkpoints loaded by CPU and GPU infer
relations:
- serves: vision:modelspec
- depends_on: story:m1-vulkan-train
- depends_on: story:m1-checkpoint-cpu
- depends_on: story:m1-vulkan-profile
- decomposes: epic:m1-vulkan-execution
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: Taskfile.yml
- confidence: inferred
  path: crates/modelspec-cli/src/infer.rs
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/checkpoint_cpu.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/checkpoint_vulkan.rs
- confidence: cited
  path: crates/modelspec-cli/tests/train_vulkan.rs
- confidence: inferred
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-docs/src/status.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/train/checkpoint.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan/sync.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_checkpoint.rs
- confidence: cited
  path: website/data/status.json
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/docs/roadmap.md
revision: 11
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T23:58:00Z", actor: "human:timo", revision: 9}
- {from: "proposed", to: "active", at: "2026-10-22T23:58:01Z", actor: "human:timo", revision: 10}
- {from: "active", to: "implemented", at: "2026-10-23T02:32:16Z", actor: "human:timo", revision: 11, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# Checkpoints of a GPU run, resume on the GPU, and `modelspec infer --backend vulkan`

## Context

DESIGN.md § 9 gate 7 asks for "GPU run equivalence within defined tolerances" after a save and
load in mid-training, and § 6.2 says a checkpoint loads "into CPU backend even if checkpoint was
made on GPU". `story:m1-checkpoint-cpu` builds the `.msckpt` format, `--checkpoint`,
`--checkpoint-every`, `--resume` and `infer` on the CPU; `story:m1-vulkan-train` runs training on
the GPU with one host wait. This story joins them, so M1 checkbox 8 holds on both backends.

DESIGN.md § 6 as `story:m1-vulkan-train` leaves it (on top of revision 0.9) has the host wait only
at the end of a run, at a requested checkpoint and at a requested read-back. Each requested
checkpoint therefore costs one host wait and one parameter read-back, and no other step does. A
Vulkan run is in the tolerance class of § 5.4, so a GPU resume is compared within the tolerance
`story:m1-vulkan-reduce-matmul` defines, not bit for bit.

## What it adds

- `train --backend vulkan` with `--checkpoint`, `--checkpoint-every` and `--resume`: the step
  plan is recorded up to the checkpoint step, which ends a command buffer; the host waits, reads the
  parameters back and writes the file, and recording continues. A resumed run uploads the
  checkpoint parameters instead of the initial ones.
- `modelspec infer <file> --backend vulkan [--device <index>] --checkpoint <path>`: uploads the
  checkpoint parameters and the synthetic batch, runs the forward plan once, waits once and prints
  the same `prediction` and `loss` lines as `--backend cpu`.
- The status page entry "Checkpoints and the compile, infer and inspect-device commands" becomes
  shipped.

## Domain relations

- Checkpoint owns ParameterRecord (`spec/domains/checkpoint.yaml:127`), matched by
  `schema_hash`, equal for Cpu and Vulkan artifacts of one file (`story:m1-kernel-plan`).
  Inferable from ess/1.

## Scope

- `crates/modelspec-runtime/src/vulkan/` (the runner), `crates/modelspec-runtime/src/train/checkpoint.rs`.
- `crates/modelspec-runtime/tests/vulkan_checkpoint.rs` (new).
- `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/src/infer.rs`,
  `crates/modelspec-cli/tests/checkpoint_vulkan.rs` (new).
- `story:m1-vulkan-profile` also edits `cli/src/train.rs` and the Vulkan runner loop; it lands
  first, and this story depends on it.
- `crates/modelspec-docs/src/status.rs`, `website/docs/roadmap.md`, and through
  `task docs-generate` `website/docs/reference/cli.md` and `website/data/status.json`.
- `AGENTS.md` (State), `CHANGELOG.md`. `CHANGELOG.md` is an append-only surface: this story adds
  its entry, and the wave coordinator orders the entries at merge.

## Acceptance

A GPU run checkpoints only where asked, resumes on the GPU within the gate-5 tolerance of the run
that never stopped, and a GPU checkpoint loads into CPU and GPU inference.

1. Operator run on the RTX 3090 (the operator starts it): `MODELSPEC_VK_VALIDATION=1 cargo test
   --release -p modelspec-runtime --features vulkan --test vulkan_checkpoint -- --ignored` passes
   with 0 validation messages.
2. In that run, `examples/mnist_mlp.mspec` with `--checkpoint run.msckpt --checkpoint-every 600`
   (1,800 steps) reads `RunStats` host waits 3 (checkpoints at 600 and 1,200, and the end) and
   writes 2 extra files; without `--checkpoint-every` it reads 1.
3. In that run, 900 steps, a checkpoint, then `--resume` for 900 steps, against 1,800 uninterrupted
   GPU steps: every final parameter passes the whole-run bound `||g - c||_2 <= 1e-4 * ||c||_2` of
   `story:m1-vulkan-reduce-matmul`, and every loss from 900 to 1,800 passes
   `GPU_ATOL + GPU_RTOL * r` with the `mse` `r` of that story. Whether the two are bit-equal is
   recorded in the verification evidence, not asserted.
4. In that run, `infer --backend cpu` on the final GPU checkpoint exits 0 and its loss passes
   `GPU_ATOL + GPU_RTOL * r` against the last loss of the GPU run; `infer --backend vulkan` on the
   same checkpoint does the same against the CPU `infer`, with 1 host wait.
5. `cargo test -p modelspec-cli --test checkpoint_vulkan`, in `task ci` and without a GPU:
   `infer --backend vulkan` with `MODELSPEC_VULKAN_LIBRARY` set to a path that does not exist exits
   1 and its stderr starts with `E-TARGET-002: `; with the variable still set and a malformed
   checkpoint it exits 1 with the E-CKPT code the CPU reader gives, because `infer` reads the
   checkpoint before it opens the loader. No test in `task ci` creates a Vulkan instance.
6. `task ci` exits 0, `task docs-check` included, with the status entry shipped and
   `infer --backend vulkan` on the CLI page.

## Out of scope

- Asynchronous checkpoint read-back that overlaps the next steps (M2).
- Optimizer slots and AdamW state.
- `infer --input` (DESIGN.md § 12 "Data interface").
