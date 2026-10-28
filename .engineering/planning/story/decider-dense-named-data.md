---
format: aep.planning-md/3
id: story:decider-dense-named-data
kind: story
status: proposed
title: Dense --data with any number of named f32 and i32 inputs and targets
summary: Bind every train declaration to NAME.f32 or NAME.i32 of a dense directory, on CPU and Vulkan, --eval included.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/fixtures/decider_small.mspec
- confidence: inferred
  path: crates/modelspec-cli/tests/train_dense_named.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-runtime/schema
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/stepwise.rs
- confidence: cited
  path: spec/domains/data.yaml
- confidence: cited
  path: website
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T12:12:29Z", actor: "human:timo", revision: 4}
---
# Dense --data with any number of named f32 and i32 inputs and targets

`depends_on: none` · first: nothing of the decider trains until this lands; first of the epic's
eight in its § Landing order.

## Outcome

`modelspec train --data DIR` on a dense directory binds every `input` and `target` declaration of
the `train` block to a file of DIR by name, `NAME.f32` for an `f32` declaration and `NAME.i32` for
an `i32` one, however many there are, and step `k` feeds block `k` of every file (the existing
rule without `order.u32`: samples `(k·B + i) mod N`; with it, `order[k·B + i]`). It works on
`--backend cpu` and `--backend vulkan`, and `--eval DIR` reads the same layout and reports the
mean loss on both backends. A program whose loss is the scalar its `forward` returns (no model
output to compare a target with) is accepted: the E-DATA-002 output-type rule does not apply to
it.

## Context

- Gap row 1. The decider program reads 19 named tensors a step, 12 of them `i32`
  (`decider.mspec` `train Joint`); `train` refuses it with E-DATA-001 ("M0 synthetic data needs
  exactly one input and one target", `crates/modelspec-runtime/src/train/data.rs:1300`), dense
  data refuses `i32` with E-DATA-004 (`data.rs:68-71`), and packed data binds only `ids`,
  `positions`, `key_mask`, `labels`, `weights` (E-DATA-005).
- Decider design § 5.3 steps 1–2 (event-token pools and decision records, § 4.2).
- Dense reading today: `data::dense`, `data::dense_eval`, `data::read_as_dense`
  (`data.rs:393`, `:455`, `:678`); the module comment (`data.rs:13-24`) states the one-input,
  one-target and `f32`-only rules this story lifts. The Vulkan stepwise runner already uploads a
  dense batch before each step (`crates/modelspec-runtime/src/vulkan/stepwise.rs:19-20`, `:223`).
  `--eval` is CPU-only today (`crates/modelspec-cli/src/train.rs:62-64`), through
  `runner::evaluate_dense` (`crates/modelspec-runtime/src/train/runner.rs:538`).
- `spec/domains/data.yaml`: `DenseTensorFile` is `f32`-only and `DenseDataDirectory` names one
  file per declaration.

## Spec first

`spec/domains/data.yaml` before any code: `DenseTensorFile` gains a `dtype`
(`modelspec.checkpoint.DType`, `F32` or `I32`) and its summary names `NAME.f32` and `NAME.i32`;
`DenseDataDirectory`'s summary drops the one-input, one-target reading and states the `--eval`
rule on both backends. Then `task spec-validate`, `task schema-generate`.

## Scope

- `spec/domains/data.yaml`, `crates/modelspec-runtime/schema/`.
- `crates/modelspec-runtime/src/train/data.rs` (binding by name, `.i32` files, E-DATA-001,
  E-DATA-002, E-DATA-004 rules), `crates/modelspec-runtime/src/train/runner.rs`
  (`evaluate_dense`), `crates/modelspec-runtime/src/vulkan/stepwise.rs` (`i32` upload, dense
  `--eval` on the device).
- `crates/modelspec-cli/src/train.rs` (`--eval` on Vulkan, help texts).
- `crates/modelspec-cli/tests/train_dense_named.rs` (new),
  `crates/modelspec-cli/tests/fixtures/decider_small.mspec` (new).
- `DESIGN.md` § 8, `crates/modelspec-docs/src/diagnostics.rs`, then `task docs-generate`
  (`website/`). One line in `CHANGELOG.md`.

## Acceptance

`cargo test -p modelspec-cli --test train_dense_named`:

- **dense-named-cpu**: a fixture with three inputs (two `i32`, one `f32`) and two targets (one
  `i32`, one `f32`) trains 10 steps on `--backend cpu` from a directory the test writes, and two
  runs from one seed print identical `step` lines.
- **dense-named-block-k**: with no `order.u32`, step `k` reads block `k` of every file: a fixture
  whose loss is a known function of the batch prints the loss the test computes for steps 0, 1
  and the wrap-around step.
- **dense-named-i32-refusals**: a missing `NAME.i32`, an `.i32` file that is not a whole number of
  samples, and files of different sample counts are refused with E-DATA-007 before the first
  step; an `f32` declaration with only `NAME.i32` present is refused naming the expected file.
- **dense-named-scalar-loss**: a fixture with exactly one target, of a type other than the model
  output's, whose loss is the scalar its `forward` returns, trains on `--backend cpu` without
  E-DATA-002; the same target compared by the loss with a model output of another type is still
  refused with E-DATA-002 before the first step. No test on `impl/decider-dense-named-data`
  (3af084f) holds this: every `train_dense_named.rs` fixture whose loss is the scalar `forward`
  returns (`NAMED`, `PROBE`, `decider_small.mspec`) declares two targets, where E-DATA-002 never
  applies. The adversary pass must cover it.
- **dense-named-eval-cpu**: `--eval DIR` prints `eval loss=<mean>` over every block; `accuracy=`
  is printed only when the block's one target is compared by the loss with the model output and
  that target is `f32`, and is absent for a loss that is the scalar `forward` returns.
- **decider-small-cpu**: `decider_small.mspec` (the decider program in its 0.21.0 workaround form,
  without `stop_gradient` and `sigreg`, at sizes small enough for a test) trains 10 steps on the
  CPU from a directory of its 19 tensors the test writes.
- `#[ignore]`d device tests, run with `--ignored`: **dense-named-vulkan** and
  **decider-small-vulkan** train the same fixtures on `--backend vulkan`, each loss within the
  DESIGN.md § 9 gate 5 tolerance of the CPU run's; **dense-named-eval-vulkan** prints the eval
  line within the same tolerance.
- `task ci` exits 0, including `task spec-validate`, `task schema-check` and `task docs-check`.

## Out of scope

- Packed directories (`binding.json`) and their E-DATA-005/006 rules: unchanged.
- `--init` on Vulkan.
- A batch format with a header, compression or `u8`/`f16` files.
- Writing the decider's batch files: the decider repository's extraction does that.
