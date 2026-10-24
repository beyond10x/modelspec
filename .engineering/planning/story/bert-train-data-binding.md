---
format: aep.planning-md/3
id: story:bert-train-data-binding
kind: story
status: draft
title: 'Train on packed MLM batches: modelspec train --data binds batch tensors by name'
summary: modelspec train --backend cpu --data binds train-block declarations to MlmBatch tensors by name, with E-DATA-005 and E-DATA-006, and writes the run record whose type story:bert-mlm-batches declares.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-mlm-batches
- depends_on: story:bert-mlm-cross-entropy
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/mlm_bag.mspec
- confidence: cited
  path: crates/modelspec-cli/tests/train_data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/optimizer.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/docs/reference/diagnostics.md
revision: 4
---
# Train on packed MLM batches: modelspec train --data binds batch tensors by name

## Context

`epic:bert-from-scratch` (In scope, Data: "sequence packing, the 80/10/10 MLM masking policy, and a
deterministic data order from the run seed"). M0 `train` has no
data file: it draws synthetic data and refuses, with `E-DATA-001`, a `train` block that does not
declare exactly one `input` and one `target`, and with `E-DATA-002` a target that is not of the
model output type (`crates/modelspec-runtime/src/train/data.rs:27-31`; DESIGN.md § 8.1). DESIGN.md
§ 8 already shows `modelspec train ... --data <file>`, and § 12 lists the data interface as open.
An MLM model reads five tensors per step, the fields of `MlmBatch` from `story:bert-mlm-batches`.

## What this story decides

- `modelspec train <file> --backend cpu --data <packed-dir>`: each `input` and `target`
  declaration of the `train` block names one `MlmBatch` tensor (`ids`, `positions`, `key_mask`,
  `labels`, `weights`), and step `k` feeds `mlm_batch` of the `Train` split at step `k` with the
  run seed and the batch size `B` read from the declared `key_mask: Tensor<f32, [B, S]>`. Without
  `--data`, M0 behaviour and its codes are unchanged.
- With `--data`, `E-DATA-001` and `E-DATA-002` do not apply. A declaration that names no batch
  tensor is refused with a new `E-DATA-005`; one whose dtype or shape differs from the batch
  tensor (an `S` other than the packed sequence length included) with a new `E-DATA-006`; both
  before anything runs. `E-DATA-003` and `E-DATA-004` keep their meaning, except that an `i32`
  declaration bound by `--data` is not refused.
- `train` prints a `data` line before the first `init` line, naming the dataset, the tokenizer and
  the masking policy, and writes them where the second settlement of `story:bert-mlm-batches`
  puts the record of a run.
- The packed directory is read once; an unreadable one exits 1 with a message naming it.

## Order on shared files

This story edits `crates/modelspec-cli/src/train.rs`, `crates/modelspec-runtime/src/train/data.rs`,
`runner.rs`, `optimizer.rs`, `DESIGN.md` and `CHANGELOG.md`, which the operator stories edit up to
`story:bert-adamw-warmup`, and reads `crates/modelspec-runtime/src/dataset/`, which
`story:bert-mlm-batches` completes. It depends on `story:bert-mlm-batches`, which depends on both
chains. `story:bert-heldout-evaluation` follows it.

## Domain relations

- A `train` declaration is a `modelspec.artifact.DataSpec` (`name`, `role`, `dtype`, `shape`) in
  `modelspec.artifact.CompiledArtifact.data`, unique by name (`distinct: {in: data, as: binding,
  by: binding.name}`, `spec/domains/artifact.yaml`). Binding a batch tensor to it by that name is
  what this story implements; the relation of a run to its `Dataset`, `Tokenizer` and
  `MaskingPolicy` is the one the second settlement of `story:bert-mlm-batches` declares, and this
  story adds none of its own.

## Scope

- `crates/modelspec-cli/src/train.rs` (`--data`, the `data` line); clap derive.
- `crates/modelspec-runtime/src/train/data.rs` (the binding, `E-DATA-005`, `E-DATA-006`),
  `runner.rs` and `optimizer.rs` (step inputs of both dtypes, one batch per step).
- The writer of the run record only, in `train.rs` and `runner.rs`: its type is declared in
  `spec/domains/` by the second settlement of `story:bert-mlm-batches`, and this story adds no type
  and no relation to `spec/`.
- `crates/modelspec-cli/tests/fixtures/mlm_bag.mspec` (new): an embedding, a tied decoder and
  `cross_entropy`, small enough to train on the fixture in seconds;
  `crates/modelspec-cli/tests/train_data.rs` (new).
- `DESIGN.md` § 6 and § 8 (`--data` on the CPU backend), § 8.1 (the new codes), § 12 (what of the
  data interface this answers) and the revision line; `website/docs/reference/` through
  `task docs-generate`; `CHANGELOG.md` one line for `--data` and the `modelspec data` commands of
  `story:bert-corpus-splits`, `story:bert-tokenizer` and `story:bert-mlm-batches`.

## Acceptance

`modelspec train --backend cpu --data` trains the fixture MLM model on batches packed from the
fixture corpus, reproducibly, and refuses every declaration that does not match a batch tensor
with a code.

Scenarios, each run by `task ci` through the command shown:

1. `cargo test -p modelspec-cli --test train_data trains`: the fixture corpus is prepared,
   tokenized and packed into a temporary directory under `target/`; `modelspec train` on
   `mlm_bag.mspec` (`steps = 200` in its `train` block) with `--data` exits 0, prints the `data`
   line first, and its last
   `step` loss is below its step-0 loss.
2. `cargo test -p modelspec-cli --test train_data reproducible`: two runs with one seed print
   identical stdout; a run with another seed prints different `step` losses.
3. `cargo test -p modelspec-cli --test train_data refusals`: a declaration named `tokens`, a
   `labels` declared `f32`, and an `S` of 8 against a sequence length of 16 exit 1 with
   `E-DATA-005`, `E-DATA-006` and `E-DATA-006`; a missing packed directory exits 1 naming it.
4. `cargo test -p modelspec-cli --test train_linear`: the M0 output of
   `examples/linear_regression.mspec` without `--data` is unchanged.
5. `grep -Eq "^\*\*Revision:\*\* [0-9.]+ \([0-9-]+\): [^;]*--data" DESIGN.md` exits 0, and
   `task docs-check` exits 0 with `E-DATA-005` and `E-DATA-006` on
   `website/docs/reference/diagnostics.md`.

## Out of scope

- `--data` on the Vulkan backend (`story:bert-vulkan-kernels-attention-loss-adamw`, after
  `epic:m1-vulkan-execution` adds `--backend vulkan`).
- A data format for non-MLM models (DESIGN.md § 12 stays open for them).
- Held-out evaluation (`story:bert-heldout-evaluation`).
