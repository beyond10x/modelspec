---
format: aep.planning-md/3
id: story:bert-vulkan-kernels-attention-loss-adamw
kind: story
status: implemented
title: Vulkan kernels for attention, cross-entropy and AdamW, and a full BERT step on the GPU
summary: SPIR-V kernels for attention, cross-entropy and the AdamW update; ten GPU steps of the hidden-32 twin, with eval lines and streamed events, match the CPU run within 1e-4 absolute plus 1e-3 relative; device runs are operator-started.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-vulkan-kernels-rowwise
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/vulkan_bert_cpu_steps.rs
- confidence: cited
  path: crates/modelspec-docs/src/status.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/
- confidence: cited
  path: crates/modelspec-runtime/src/train/eval.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/
- confidence: cited
  path: crates/modelspec-runtime/tests/spirv_bert_attention_loss_adamw.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/vulkan_bert_attention_loss_adamw.rs
- confidence: cited
  path: website/data/status.json
- confidence: cited
  path: website/docs/reference/
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T22:35:38Z", actor: "human:timo", revision: 5}
- {from: "proposed", to: "active", at: "2026-10-25T22:35:38Z", actor: "human:timo", revision: 6}
- {from: "active", to: "implemented", at: "2026-10-26T03:05:50Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# Vulkan kernels for attention, cross-entropy and AdamW, and a full BERT step on the GPU

## Context

`epic:bert-from-scratch` (In scope, Execution: "the GPU backend (epic:m1-vulkan-execution) for the
full run"; Acceptance: "trains on the GPU backend; held-out MLM loss falls below a target set in
decomposition from a reference run"; Outcome: "the run is watched in the training observatory").
After `story:bert-vulkan-kernels-rowwise` the operators left without a kernel are the attention
family, cross-entropy and the AdamW update with its internal `Sqrt` and `Div`. With them, a whole
training step of the hidden-32 twin of `story:bert-tiny-cpu-model` runs on the GPU and can be
compared with the CPU oracle, which is DESIGN.md § 9 gate 5 for a complete graph. This story also
makes the two things the GPU run needs besides training work on `--backend vulkan`: held-out
evaluation (`story:bert-heldout-evaluation` builds it on the CPU runner) and the event stream of
the observatory (`epic:training-observatory` is CPU-only: "the observatory must run against
`--backend cpu`").

## What this story decides

- SPIR-V kernels, behind the `vulkan` feature, for `Attention`, `AttentionGradQ`,
  `AttentionGradK`, `AttentionGradV`, `CrossEntropy`, `CrossEntropyGrad`, `Sqrt`, `Div`, and the
  AdamW `Optimizer` subgraph of `story:bert-adamw-warmup` with its moment state in device buffers.
  Naive kernels first (DESIGN.md § 5.2: correctness before speed), no flash-style fusion.
- `--backend vulkan --data` trains: the batch tensors of `story:bert-train-data-binding` and the
  per-step scalars and dropout tokens are uploaded each step.
- `--backend vulkan --data --eval-every <n>` evaluates as `story:bert-heldout-evaluation` defines,
  at the same steps and over the same `HeldOut` batches with dropout off, running the forward
  program on the device, and prints the same `eval <k> loss <v> accuracy <a>` lines.
- The Vulkan path of `runner.rs` calls the per-step observer of
  `story:observatory-live-event-stream`, so the serve flag of that story and the record flag of
  `story:observatory-run-recording-replay` work with `--backend vulkan`: `RunStarted`, one
  `StepCompleted` per recorded loss with the loss and, at evaluated steps, the samples
  `heldout_mlm_loss` and `heldout_mlm_accuracy`, then `RunFinished` or `RunFailed`. The parameter,
  activation and saliency data of the later observatory stories are not produced on a Vulkan run.
- Reductions keep a fixed order under a deterministic compile, as in
  `story:bert-vulkan-kernels-rowwise`.

## Tolerances

- Kernels: the gate-5 rule of `story:bert-vulkan-kernels-rowwise`, `|g - c| <= 1e-5 + 1e-4 * |c|`
  per element (`GPU_ATOL`, `GPU_RTOL`), with every reduction in a device test at most 512 terms
  long, so cross-entropy tests use `V <= 512`.
- A run of several training steps compounds kernel differences, so a whole run has its own rule:
  every printed `step` and `eval` loss and every final parameter element satisfies
  `|g - c| <= 1e-4 + 1e-3 * |c|` against the `--backend cpu` run (`GPU_RUN_ATOL = 1e-4`,
  `GPU_RUN_RTOL = 1e-3`, ten times the kernel rule). A held-out accuracy may differ from the CPU
  one by at most `1 / M`, `M` the number of selected `HeldOut` positions, because one argmax may
  flip at a near tie.
- These two constants and the accuracy rule are written into DESIGN.md § 9 beside the kernel
  ones, with a revision-line entry, before any device scenario runs.

## Order on shared files

This story edits the Vulkan and SPIR-V modules of `crates/modelspec-runtime/`, the Vulkan path of
`crates/modelspec-runtime/src/train/runner.rs` and `src/train/eval.rs`,
`crates/modelspec-cli/src/train.rs`, `DESIGN.md` § 8, § 9 and the revision line,
`crates/modelspec-docs/src/status.rs`, `website/data/status.json` and `CHANGELOG.md` after
`story:bert-vulkan-kernels-rowwise`, so it depends on it; `eval.rs` comes from
`story:bert-heldout-evaluation` and the observer in `runner.rs` from
`story:observatory-live-event-stream`, both ancestors through that edge.
`story:bert-tiny-gpu-pretraining` follows it.

## Domain relations

- Each kernel is a `modelspec.artifact.KernelModule` owned by its `CompiledArtifact` (relation
  `kernels`, `kind: owns`, `spec/domains/artifact.yaml`).
- A Vulkan run references its device: `modelspec.training.TrainingRun.device`
  (`kind: references`, target `modelspec.device.DeviceProfile`, `spec/domains/training.yaml`),
  present exactly on a Vulkan run (the summary of `device_profile_id`).
- A streamed Vulkan run owns its events as a CPU run does: `modelspec.training.TrainingRun.events`
  (`kind: owns`, target `modelspec.observatory.RunEvent`, `spec/domains/training.yaml`); the two
  samples are entries of `modelspec.observatory.StepCompleted.metrics`, distinct by name
  (`spec/domains/observatory.yaml`).

## Scope

- `crates/modelspec-runtime/src/spirv/` and `src/vulkan/` (the files `epic:m1-vulkan-execution`
  creates there): the eight kernels and the AdamW update.
- `crates/modelspec-runtime/src/train/runner.rs` (Vulkan batches, the observer on the Vulkan
  path) and `src/train/eval.rs` (evaluation on the device).
- `crates/modelspec-cli/src/train.rs` (`--data`, `--eval-every` and the serve and record flags
  accepted with `--backend vulkan`).
- `crates/modelspec-runtime/tests/spirv_bert_attention_loss_adamw.rs`,
  `crates/modelspec-runtime/tests/vulkan_bert_attention_loss_adamw.rs` and
  `crates/modelspec-cli/tests/vulkan_bert_cpu_steps.rs` (new).
- `DESIGN.md` § 8 (evaluation and streaming on the Vulkan backend), § 9 (the run tolerances) and
  the revision line; `website/docs/reference/` through `task docs-generate`.
- `crates/modelspec-docs/src/status.rs` and `website/data/status.json`; `CHANGELOG.md` one line.

## Acceptance

On a Vulkan device, the attention, cross-entropy and AdamW kernels are within
`1e-5 + 1e-4 * |c|` of the CPU oracle, and ten training steps of `bert_cpu.mspec` with `--data`
and `--eval-every 5` on the GPU print `step` and `eval` losses and final parameters within
`1e-4 + 1e-3 * |c|` of the CPU run, accuracies within `1 / M`, and stream them as schema-valid
events.

Scenarios:

1. `grep -Eq "^\*\*Revision:\*\* [0-9.]+ \([0-9-]+\): [^;]*run tolerance" DESIGN.md` exits 0, and
   DESIGN.md § 9 states `1e-4` absolute and `1e-3` relative for a run and the `1 / M` accuracy
   rule; run by hand before any device scenario.
2. `cargo test -p modelspec-runtime --features vulkan --test spirv_bert_attention_loss_adamw`:
   every new module passes `spirv-val`. It needs no device; it is part of `task ci` only if
   `epic:m1-vulkan-execution` puts the `vulkan` feature and `spirv-val` into `task ci`, which I do
   not know yet.
3. Device run, started by the operator on a machine with a Vulkan device, not part of `task ci`:
   `cargo test -p modelspec-runtime --features vulkan --test vulkan_bert_attention_loss_adamw --
   --ignored`: with a partial key mask and `heads = 2`, and for cross-entropy with `V = 512` and
   zero weights, every forward and gradient element satisfies `|g - c| <= 1e-5 + 1e-4 * |c|`;
   three AdamW steps on identical gradients satisfy the same rule for parameters, `m` and `v`.
4. Device run, as scenario 3: `cargo test -p modelspec-cli --features vulkan --test
   vulkan_bert_cpu_steps steps -- --ignored`: `modelspec train --backend vulkan --data` on
   `bert_cpu.mspec` for 10 steps prints `step` losses and final parameters within
   `1e-4 + 1e-3 * |c|` of the `--backend cpu` run, and two GPU runs with one seed print identical
   stdout.
5. Device run, as scenario 3: `cargo test -p modelspec-cli --features vulkan --test
   vulkan_bert_cpu_steps heldout_eval -- --ignored`: with `--eval-every 5`, the GPU run prints
   `eval` lines for steps 0, 5 and 10, each loss within `1e-4 + 1e-3 * |c|` of the line of the
   `--backend cpu` run and each accuracy within `1 / M` of it.
6. Device run, as scenario 3: `cargo test -p modelspec-cli --features vulkan --test
   vulkan_bert_cpu_steps streamed -- --ignored`: with `--eval-every 5` and the serve and record
   flags, every event a client reads validates against the observatory JSON Schema, the loss of
   each `StepCompleted` equals its `step` line as `f32` shortest round-trip text, the events of
   steps 0, 5 and 10 carry `heldout_mlm_loss` and `heldout_mlm_accuracy` equal to the `eval`
   lines and the others carry neither, and replaying the recorded file serves the same events.
7. `task ci` exits 0 without a Vulkan device, and `task docs-check` exits 0.

## Out of scope

- Fused attention, cooperative matrices, f16 (DESIGN.md row M3 and § 5.2, later).
- Parameter statistics, activation statistics and saliency on a Vulkan run
  (`story:observatory-parameter-statistics`, `story:observatory-activation-statistics` and
  `story:observatory-saliency-gradient-x-input` build them on the CPU runner).
- The BERT-tiny run (`story:bert-tiny-gpu-pretraining`).
