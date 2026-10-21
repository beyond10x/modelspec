---
format: aep.planning-md/3
id: story:bert-heldout-evaluation
kind: story
status: draft
title: Held-out MLM loss and accuracy during training, printed and streamed as metrics
summary: 'Adds --eval-every: held-out MLM loss and accuracy over the HeldOut split with dropout off, printed as eval lines and streamed as two StepCompleted metric samples.'
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-train-data-binding
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/mlm_bag_dropout.mspec
- confidence: cited
  path: crates/modelspec-cli/tests/train_eval.rs
- confidence: cited
  path: crates/modelspec-observatory/
- confidence: cited
  path: crates/modelspec-runtime/src/train/eval.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: website/docs/reference/
revision: 3
---
# Held-out MLM loss and accuracy during training, printed and streamed as metrics

## Context

`epic:bert-from-scratch` (In scope, Evaluation: "MLM loss and accuracy on a held-out split";
Acceptance: "held-out MLM loss falls below a target"). `modelspec.observatory.MetricSample`
is "One named metric value at a step, such as MLM accuracy" and `StepCompleted.metrics` holds them,
names unique (`spec/domains/observatory.yaml`). `story:observatory-ui-live-loss-chart` draws a
metric from a fixture whose events carry one; no producer fills `metrics` yet.

## What this story decides

- `modelspec train ... --data <dir> --eval-every <n>`: at step 0, every `n` steps and at the last
  step, the forward program on the current parameters, dropout off (`story:bert-dropout`), runs
  over every full batch of the `HeldOut` split in stored order, with masking drawn the same way at
  every evaluation so that values are comparable over a run.
- Held-out loss: the sum over batches of `sum weights * nll` divided by the sum of all weights, the
  same quantity `cross_entropy` computes on one batch. Accuracy: the share of selected positions
  whose highest logit is the label, ties going to the lowest id. The logits are the value the
  model forward returns, read as the intermediate `prediction` as M0 synthetic data reads it
  (inferred from the module doc of `crates/modelspec-runtime/src/train/data.rs`).
- `train` prints `eval <k> loss <v> accuracy <a>` after the `step <k>` line of an evaluated step,
  and, when the run streams events, the `StepCompleted` of that step carries
  `MetricSample { name: "heldout_mlm_loss" }` and `"heldout_mlm_accuracy"`; other steps carry
  neither.
- `--eval-every` without `--data` is a usage error, exit 2; `n = 0` likewise.

## Order on shared files

This story edits `crates/modelspec-cli/src/train.rs`, `crates/modelspec-runtime/src/train/runner.rs`,
`DESIGN.md` § 8 and `CHANGELOG.md` after `story:bert-train-data-binding`, so it depends on it, and
`crates/modelspec-observatory/` (the metric samples), last edited by the observatory stories that
`story:bert-embedding-lookup` follows. `story:bert-tiny-cpu-model` follows it.

## Domain relations

- `modelspec.observatory.StepCompleted.metrics` is a field `List<modelspec.observatory.MetricSample>`
  with `distinct: {in: metrics, as: sample, by: sample.name}` (`spec/domains/observatory.yaml`), so
  the two samples have distinct names and belong to their step event.
- The held-out split is the `DatasetSplit` with `role: HeldOut` of the `Dataset`
  (`modelspec.data.SplitRole`, "MLM loss and accuracy are evaluated on the held-out split",
  `spec/domains/data.yaml`).

## Scope

- `crates/modelspec-cli/src/train.rs` (`--eval-every`, the `eval` line); clap derive.
- `crates/modelspec-runtime/src/train/runner.rs` and a new `src/train/eval.rs`.
- `crates/modelspec-observatory/`: the two samples in `StepCompleted.metrics`.
- `crates/modelspec-cli/tests/train_eval.rs` (new), on `mlm_bag.mspec` of
  `story:bert-train-data-binding`, and `crates/modelspec-cli/tests/fixtures/mlm_bag_dropout.mspec`
  (new): `mlm_bag.mspec` with one `dropout` call after the embedding.
- `DESIGN.md` § 8 (the `eval` line) and the revision line; `website/docs/reference/` through
  `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

A `--data --eval-every` run prints held-out MLM loss and accuracy at the evaluated steps, equal to
what the test computes from the printed parameters, and streams them as two metric samples.

Scenarios, each run by `task ci` through the command shown:

1. `cargo test -p modelspec-cli --test train_eval values`: `modelspec train` on `mlm_bag.mspec` (`steps = 200`
   in its `train` block) with `--eval-every 100` prints `eval` lines for steps 0, 100 and 200; the
   step-200 loss and
   accuracy equal those the test computes over the fixture `HeldOut` batches from the final
   printed parameters, within `1e-6`.
2. `cargo test -p modelspec-cli --test train_eval dropout_off`: `mlm_bag_dropout.mspec` is
   `mlm_bag.mspec` with one `dropout(..., p = 0.1)` call after the embedding and nothing else
   changed. Run with one seed and `--eval-every 100`, the two files print identical `eval 0` lines
   bit for bit, because their initial parameters are equal (DESIGN.md § 2.3: drawn from the seed
   and the canonical path, and `dropout` declares no parameter), while their `step 0` lines differ,
   because dropout is on in the training step.
3. `cargo test -p modelspec-cli --test train_eval accuracy_ties`: for a fixture whose logits tie,
   the reported accuracy counts the lowest id as the prediction.
4. `cargo test -p modelspec-cli --test train_eval streamed`: with the serve flag of
   `story:observatory-live-event-stream`, the `StepCompleted` events of steps 0, 100 and 200 carry
   `heldout_mlm_loss` and `heldout_mlm_accuracy` equal to the printed values, the other events
   carry neither, and every event validates against the observatory JSON Schema.
5. `cargo test -p modelspec-cli --test train_eval usage`: `--eval-every 10` without `--data` and
   `--eval-every 0` exit 2.
6. `grep -Eq "^\*\*Revision:\*\* [0-9.]+ \([0-9-]+\): [^;]*eval" DESIGN.md` and
   `task docs-check` exit 0.

## Out of scope

- A downstream probe (optional in the epic; not drafted).
- Perplexity or other metrics.
- Evaluation and metric samples on `--backend vulkan` (`story:bert-vulkan-kernels-attention-loss-adamw`).
