---
format: aep.planning-md/3
id: story:train-data-dense-tensors
kind: story
status: implemented
title: 'Train on dense f32 tensors from files: modelspec train --data, --init, --eval'
summary: modelspec train reads dense f32 input and target tensors, optional sample order and initial params from a directory, and prints test loss and accuracy, so the MNIST benchmark runs through the release CLI
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
scope:
- confidence: inferred
  path: CHANGELOG.md
- confidence: inferred
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/adversary_dense_1.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/train_dense_data.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/train/init.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: inferred
  path: spec/domains/data.yaml
- confidence: inferred
  path: website/docs/reference/cli.md
- confidence: inferred
  path: website/docs/reference/diagnostics.md
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T03:24:51Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-10-22T03:24:51Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-10-22T06:09:00Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"test_result":2,"review_outcome":1}}}
---
# Train on dense f32 tensors from files: modelspec train --data, --init, --eval

## Context

`epic:pytorch-10x-mnist` measures the 784-64-10 MNIST example against PyTorch, and
`verification-report:mnist-benchmark-0-1-1` could only run it through a custom Rust adapter over
the library API, because `modelspec train` has no `--data`: it draws synthetic data
(`crates/modelspec-runtime/src/train/data.rs`, DESIGN.md § 8.1). `story:bert-train-data-binding`
adds `--data` for packed MLM batches, but it depends on `story:bert-mlm-batches`, which depends on
the tokenizer and AdamW chain, all draft.

Split (2026-10-22, approved by the coordinating session under the operator's grant): the MNIST
lane does not wait on the BERT chain. This story adds `--data` for dense `f32` tensors only.
`story:bert-train-data-binding` keeps the MLM batch binding and builds on this flag and file
reader; its `depends_on` is unchanged, and the codes it reserves (`E-DATA-005`, `E-DATA-006`) are
left free here.

## What this story decides

- `modelspec train <file> --backend cpu --data <dir>`: each `input` and `target` declaration
  `name: Tensor<f32, [B, d1, …]>` of the `train` block reads `<dir>/<name>.f32`, row-major
  little-endian `f32`, one sample per `d1 × …` values; `N` samples is the file length divided by
  the sample size. Step `k` feeds samples `order[k·B .. (k+1)·B]` when `<dir>/order.u32`
  (little-endian `u32` sample indices) exists, and samples `(k·B + i) mod N` otherwise.
- `--init <dir>`: each `param` named `p` with a file `<dir>/p.f32` starts from that file instead
  of its initializer; a param with no file keeps its initializer.
- `--eval <dir>`: after the last step, the forward program runs over every sample of `<dir>` in
  batches of `B`, and `train` prints one line `eval loss=<mean loss> accuracy=<fraction>`, where a
  sample counts as correct when the argmax of the output row equals the argmax of the target row.
- Without these flags, M0 behaviour, output and codes are unchanged.
- New codes, each before the first step: `E-DATA-007`, a file is missing, unreadable, or its
  length is not a whole number of samples (or, for `--init`, not the param's size);
  `E-DATA-008`, `order.u32` holds fewer than `steps · B` indices or an index `≥ N`;
  `E-DATA-009`, an `--eval` sample count that is not a multiple of `B`.
- The directory layout is a file-on-disk contract and is typed in `spec/` (AGENTS.md,
  spec-first) before code reads it.

## Scope

- `spec/domains/data.yaml` or `spec/domains/training.yaml`: the dense data directory type.
- `crates/modelspec-cli/src/train.rs`: `--data`, `--init`, `--eval`, the `eval` line.
- `crates/modelspec-runtime/src/train/data.rs` (the file reader and the codes),
  `crates/modelspec-runtime/src/train/runner.rs`, `crates/modelspec-runtime/src/train/init.rs`.
- `crates/modelspec-cli/tests/train_dense_data.rs` (new) with small fixtures written by the test.
- `DESIGN.md` § 8 and § 8.1, the revision line; `website/docs/reference/` through
  `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`modelspec train --data` trains the benchmark example from files and reports its accuracy.

1. `ess specify validate --path .` from `spec/` exits 0 and the dense layout type is declared.
2. `cargo test -p modelspec-cli --test train_dense_data`: a 4-sample fixture trains with and
   without `order.u32` and the step-0 loss equals the loss computed from the fixture by hand;
   `--init` replaces exactly the named params; each of `E-DATA-007`, `E-DATA-008`, `E-DATA-009`
   is produced, exits 1, and no `step` line is printed.
3. Without `--data`, every existing `train` golden passes unchanged.
4. Run by hand, outside the repository: the release binary trains the benchmark's
   `model.mspec` with `--data` (train images, one-hot targets, `order.u32`), `--init` (the saved
   `w1`, `w2`) and `--eval` (the test set); `accuracy` is 0.9088 and `loss` is within 1e-6 of
   0.026219031689688563, as in `verification-report:mnist-benchmark-0-1-1`. The wall time of the
   1,800 steps is recorded.
5. `task ci` exits 0, `task docs-check` included; no Python, PyTorch or CUDA enters the tree.

## Out of scope

- Packed MLM batches, `i32` inputs, tokenizers: `story:bert-train-data-binding`.
- IDX or Parquet readers; the preparation of MNIST files stays outside the repository.
- The GPU backend: `story:m1-vulkan-train`.
