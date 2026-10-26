---
format: aep.planning-md/3
id: story:bert-tiny-gpu-pretraining
kind: story
status: implemented
title: Pre-train BERT-tiny on the GPU until held-out MLM loss is below the unigram baseline
summary: Operator-run GPU pre-training of examples/bert_tiny.mspec on the settled corpus, watched in the observatory and recorded, ending below the held-out unigram baseline, with the curve and margin recorded as evidence.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-vulkan-kernels-attention-loss-adamw
scope:
- confidence: cited
  path: examples/bert_tiny.mspec
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-26T03:57:20Z", actor: "human:timo", revision: 6}
- {from: "proposed", to: "active", at: "2026-10-26T03:57:21Z", actor: "human:timo", revision: 7}
- {from: "active", to: "implemented", at: "2026-10-26T08:15:47Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"test_result":1,"metric_observation":1,"review_outcome":1}}}
---
# Pre-train BERT-tiny on the GPU until held-out MLM loss is below the unigram baseline

## Context

`epic:bert-from-scratch` (Outcome: "its loss falls to a stated target; the run is watched in the
training observatory"; Acceptance: "A BERT-tiny `.mspec` passes `modelspec check` and trains on the
GPU backend; held-out MLM loss falls below a target set in decomposition from a reference run").
Everything this run needs exists after `story:bert-vulkan-kernels-attention-loss-adamw`: the model
`examples/bert_tiny.mspec` (`story:bert-tiny-cpu-model`), the data commands with the corpus the
operator settled in `story:bert-corpus-splits`, `--data`, and on `--backend vulkan` the kernels,
`--eval-every` and the serve and record flags of the observatory, which that story makes work on
the GPU.

## The target, and why it is not a number yet

The epic asks for a target "set in decomposition from a reference run". No reference run of
BERT-tiny can be made before this story: the GPU path does not exist until
`epic:m1-vulkan-execution` and the two kernel stories, and a CPU run of 4.4M parameters on the
reference interpreter is not a practical reference. The target set here is therefore:

- the final held-out MLM loss is strictly below the held-out unigram baseline that
  `modelspec data pack` records for the settled corpus (`story:bert-mlm-batches`), the loss of a
  model that predicts every masked token from train-split frequencies alone; and
- the held-out loss curve of the run, its final held-out accuracy and the margin below the baseline
  are recorded, so the operator can set a tighter numeric target for later runs from this one.

## What this story decides

- An agent runs, on the local Vulkan device with no warm-up step: `modelspec data prepare`,
  `modelspec data tokenizer --vocab-size 8192` (the vocabulary D19 and D71 fixed; coordinator decision D75), `modelspec data pack --seq-len 128` on the
  settled corpus, then `modelspec train examples/bert_tiny.mspec --backend vulkan --data <dir>
  --eval-every <n>` with the serve flag of `story:observatory-live-event-stream` and the record
  flag of `story:observatory-run-recording-replay`. The step count and `n` are set in the `train`
  block and on the command line by the agent that runs it and recorded with the result.
- The run is watched in the training observatory: the UI of `story:observatory-ui-live-loss-chart`
  is opened on the served stream while the run goes; it shows the
  training loss and the two held-out samples. The recorded file is the checkable trace of what the
  observatory showed.
- The record is the full stdout of the run, the commands with their versions (the commit, the
  `modelspec` version), the device line, the baseline, the final held-out loss and accuracy, and
  the recorded event file, attached as evidence of this story through the `aep` CLI, never by
  hand.

## Order on shared files

This story changes no file of the repository unless the run shows a defect, which becomes its own
story; it may change `steps` and the `adamw` arguments in `examples/bert_tiny.mspec`, which
`story:bert-tiny-cpu-model` created. It depends on `story:bert-vulkan-kernels-attention-loss-adamw`.

## Domain relations

- The run is a `modelspec.training.TrainingRun` with `artifact` (`kind: references`,
  `modelspec.artifact.CompiledArtifact`) and `device` (`kind: references`,
  `modelspec.device.DeviceProfile`) (`spec/domains/training.yaml`), and owns its `events`
  (`kind: owns`, `modelspec.observatory.RunEvent`, same file), which this run streams and records.
  Its dataset, tokenizer and masking policy are recorded where the second settlement of
  `story:bert-mlm-batches` puts them.

## Scope

- No code. The commands above; `examples/bert_tiny.mspec` only for `steps` and the
  optimizer arguments; the evidence record with the recorded event file.

## Acceptance

A GPU pre-training run of `examples/bert_tiny.mspec` on the settled public corpus, watched in the
observatory, ends with a held-out MLM loss strictly below the held-out unigram baseline of that
corpus, and its record names the loss curve, the final accuracy and the margin and holds the
recorded events.

Scenarios, each a device run started by an agent on the local GPU, none part of `task ci`:

1. `modelspec train examples/bert_tiny.mspec --backend vulkan --data <packed> --eval-every <n>`
   with the serve and record flags exits 0 and prints `eval` lines (the device is the
   `inspect-device --backend vulkan` output recorded beside the run: `train` prints no device
   line, coordinator decision D76), and its last `eval` loss is below the `baseline` value in the record of `<packed>`.
2. The `eval` loss falls across the run: the last `eval` loss is below the step-0 `eval` loss, and
   the record shows every `eval` line.
3. Replaying the recorded file with the replay verb or flag of
   `story:observatory-run-recording-replay` exits 0; its events validate against the observatory
   JSON Schema, there is one `StepCompleted` per step of the run (step 0 to the last; `train`
   prints a `step` line every 50 steps, and each printed line's loss equals its event's; decision
   D76), and the `heldout_mlm_loss` of its last evaluated step equals the last `eval` loss as `f32` shortest round-trip text.
4. The evidence record of this story holds the stdout, the commands, the commit, the device, the
   baseline, the final held-out loss and accuracy, the margin and the recorded event file.

## Out of scope

- A tighter numeric target, BERT-mini and BERT-small, a downstream probe (optional in the epic;
  not drafted).
- Parameter statistics, activation statistics and saliency of this run: the observatory stories
  that produce them run on the CPU only, and this run is on the GPU.
- Resuming from a checkpoint (AdamW moments in checkpoints are not drafted).
