# Checkpoints of a GPU run, resume on the GPU, and `modelspec infer --backend vulkan`

## Context

DESIGN.md § 9 gate 7 asks for "GPU run equivalence within defined tolerances" after a save and
load in mid-training, and § 6.2 says a checkpoint loads "into CPU backend even if checkpoint was
made on GPU". `story:m1-checkpoint-cpu` builds the `.msckpt` format, `--checkpoint`,
`--checkpoint-every`, `--resume` and `infer` on the CPU; `story:m1-vulkan-train` runs training on
the GPU with one host wait. This story joins them, so M1 checkbox 8 holds on both backends.

The 10x path of `epic:pytorch-10x-mnist` reads results back only at the end or at a requested
checkpoint. Each requested checkpoint therefore costs one host wait and one parameter read-back,
and no other step does.

## What it adds

- `train --backend vulkan` with `--checkpoint`, `--checkpoint-every` and `--resume`: the step
  plan is recorded up to the checkpoint step, the host waits, reads the parameters back and writes
  the file, and recording continues. A resumed run uploads the checkpoint parameters instead of
  the initial ones.
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
- `crates/modelspec-docs/src/status.rs`, `website/docs/roadmap.md`, `website/data/` through
  `task docs-generate`. `AGENTS.md` (State), `CHANGELOG.md`.

## Acceptance

A GPU run checkpoints only where asked, resumes on the GPU within the gate-5 tolerance of the run
that never stopped, and a GPU checkpoint loads into CPU and GPU inference.

1. Operator run on the RTX 3090 (the operator starts it): `MODELSPEC_VK_VALIDATION=1 cargo test
   --release -p modelspec-runtime --features vulkan --test vulkan_checkpoint -- --ignored` passes
   with 0 validation messages.
2. In that run, `examples/mnist_mlp.mspec` with `--checkpoint run.msckpt --checkpoint-every 600`
   (1,800 steps) reads
   `RunStats` host waits 3 (checkpoints at 600 and 1,200, and the end) and writes 2 extra files;
   without `--checkpoint-every` it reads 1.
3. In that run, 900 steps, a checkpoint, then `--resume` for 900 steps, against 1,800 uninterrupted
   GPU steps: every final parameter satisfies `||g - c||_2 <= 1e-4 * ||c||_2` and every loss from
   900 to 1,800 passes `GPU_ATOL + GPU_RTOL * |c|`. Whether the two are bit-equal is recorded in the
   verification evidence, not asserted.
4. In that run, `infer --backend cpu` on the final GPU checkpoint exits 0 and its loss passes
   `GPU_ATOL + GPU_RTOL * |c|` against the GPU run last loss; `infer --backend vulkan` on the same
   checkpoint does the same against the CPU `infer`, with 1 host wait.
5. `cargo test -p modelspec-cli --test checkpoint_vulkan`, in `task ci` and without a GPU:
   `infer --backend vulkan` with `MODELSPEC_VULKAN_LIBRARY` set to a missing path exits 1 with
   E-TARGET-002, and with a malformed checkpoint exits 1 with the E-CKPT code the CPU reader gives.
6. `task ci` exits 0, `task docs-check` included, with the status entry shipped.

## Out of scope

- Asynchronous checkpoint read-back that overlaps the next steps (M2).
- Optimizer slots and AdamW state.
- `infer --input` (DESIGN.md § 12 "Data interface").
