---
format: aep.planning-md/3
id: story:vulkan-eval-download-per-batch
kind: story
status: active
title: The stepwise Vulkan run evaluates held-out batches through a one-batch download buffer
summary: 'Fixes the --eval-every refusal on examples/bert_tiny.mspec over the packed corpus: the stepwise run sized its download buffer as every HeldOut batch''s logits at once (9.1 GB over maxStorageBufferRange); it now evaluates one batch per command buffer through a one-prediction buffer.'
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- blocks: story:bert-tiny-gpu-pretraining
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/stepwise.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/
- confidence: cited
  path: website/docs/reference/cli.md
revision: 11
transitions:
- {from: "draft", to: "proposed", at: "2026-10-26T04:09:06Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-26T04:09:06Z", actor: "human:timo", revision: 3}
---
# The stepwise Vulkan run evaluates held-out batches through a one-batch download buffer

## Context

`story:bert-tiny-gpu-pretraining`'s first device run of `examples/bert_tiny.mspec` with
`--eval-every 50` on the packed WikiText-103 corpus (2,189 HeldOut sequences, 68 full batches of
32) exited 1 before step 0 with `9126805508 bytes exceed maxStorageBufferRange 4294967295`
(coordinator decision D76). `crates/modelspec-runtime/src/vulkan/stepwise.rs:388` sizes one
download buffer as `4 + eval_batches * prediction.size_bytes`, the logits of every HeldOut batch
of an evaluated step at once: 68 × 134,217,728 bytes of `[4096, 8192]` f32 logits. The buffer is
created before step 0, so no `--eval-every` value runs on that model and corpus, whatever the
step. The hidden-32 twin's tests never reached the limit because their HeldOut split is small.

## What this story decides

- The stepwise run (`vulkan::stepwise`) evaluates the HeldOut split one batch at a time: each
  evaluated batch is its own command buffer (copy the batch in, run the loss actions, copy the
  prediction out, one host wait), read into the same download buffer of one prediction, and
  tallied on the host as the CPU path tallies (`train::eval::Tally`, in batch order), so the
  `eval` loss and accuracy, the printed lines and the `StepCompleted` metric samples are what
  they were. The download buffer's size no longer depends on the number of HeldOut batches:
  `4 + prediction.size_bytes` at most, whatever the split.
- The sizing is a function the tests can call without a device, and a device test evaluates a
  packed directory whose HeldOut split has more batches than the twin's fixtures (at least 64
  full batches, built by the test from the synthetic corpus of `tests/common/markov_corpus.rs`)
  and compares its `eval` lines with the CPU run's within the run tolerance of DESIGN.md § 9.
- The counters a `--stats` run prints count the evaluation's host waits and submits as they
  happen (one per evaluated batch), and DESIGN.md § 8 says the evaluation of a stepwise run
  takes one host wait per HeldOut batch; the sentence of 0.53 that put the evaluation in the
  step's command buffer with no wait of its own goes, in a 0.54 revision entry. The `--eval-every`
  help and the generated CLI page say the same.
- Downloading every batch's logits to tally them on the host is the correctness-first form
  (DESIGN.md § 5.2). Tallying on the device (the loss sum from the `CrossEntropy` kernel, the
  accuracy from a kernel that counts argmax matches, two floats down per batch) is
  `story:vulkan-eval-metrics-on-device`, drafted beside this story and not part of it.

## Order on shared files

This story edits `crates/modelspec-runtime/src/vulkan/stepwise.rs`, its tests, DESIGN.md § 8,
`crates/modelspec-cli/src/train.rs` (the `--eval-every` help) and the generated CLI page, and
CHANGELOG.md. It runs alone, before `story:bert-tiny-gpu-pretraining` resumes; nothing else edits
these files now.

## Domain relations

None new: the run stays a `modelspec.training.TrainingRun` and its evaluation the HeldOut
`modelspec.data.DatasetSplit` (`spec/domains/training.yaml`, `spec/domains/data.yaml`).

## Scope

- `crates/modelspec-runtime/src/vulkan/stepwise.rs`: the per-batch evaluation command buffers,
  the one-prediction download buffer, the sizing function.
- `crates/modelspec-runtime/tests/` or `crates/modelspec-cli/tests/`: the sizing test and the
  device test with a large HeldOut split (`#[ignore]` as the other device tests are).
- `crates/modelspec-cli/src/train.rs`, `website/docs/reference/cli.md` (generated): the
  `--eval-every` help.
- DESIGN.md § 8 and its revision line (0.54); CHANGELOG.md Fixed.

## Acceptance

`modelspec train <copy of examples/bert_tiny.mspec with steps = 100 and warmup_steps = 10>
--backend vulkan --data <packed WikiText-103 at seq-len 128> --eval-every 50` on the RTX 3090
exits 0 and prints `eval 0 …`, `eval 50 …` and `eval 100 …` lines (the example itself refuses
`--steps 100` under its 1,000 warm-up steps, and step 0 is a multiple of 50), and the hidden-32 twin's `eval` lines on the device still match the CPU's
within the run tolerance.

Scenarios:

1. The sizing function, given a prediction of `[4096, 8192]` f32 and 68 batches, answers
   `4 + 134217728`, under `maxStorageBufferRange` of the RTX 3090; `vulkan_bert_cpu_steps::steps`
   stays green.
2. A device run of the twin over a packed directory with at least 64 full HeldOut batches and
   `--eval-every 5 --steps 10` exits 0 and its `eval` lines match the CPU run's within the run
   tolerance; a `--stats` run of it reports the evaluation's host waits (one per batch per
   evaluated step).
3. The 100-step copy of `examples/bert_tiny.mspec` with `--eval-every 50` on the packed corpus
   exits 0 with three `eval` lines (a device run by the implementor, quoted in the report, not in `task ci`).

## Out of scope

- Device-side tallying (`story:vulkan-eval-metrics-on-device`).
- A smaller or sampled held-out evaluation.
