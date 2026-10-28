---
format: aep.planning-md/3
id: story:decider-infer-on-data
kind: story
status: draft
title: infer --data DIR --function NAME --out FILE
summary: Score a dense directory with a named model function and write its rows.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
- depends_on: story:decider-dense-named-data
- depends_on: story:decider-model-function-calls
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/infer.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/infer_data.rs
- confidence: cited
  path: crates/modelspec-core/src/compile.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/cli.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-runtime/schema
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: spec/domains/data.yaml
- confidence: cited
  path: website
revision: 5
---
# infer --data DIR --function NAME --out FILE: score a dense directory with a named model function

`depends_on: story:decider-dense-named-data` (it reads DIR as S1's dense reader does) and
`story:decider-model-function-calls`; lands last of the epic's eight, see the epic's § Landing
order.

## Outcome

`modelspec infer <file> --backend cpu|vulkan --checkpoint <path> --data DIR --function NAME --out
FILE` runs the model function `NAME` with the checkpoint's parameters over every block of `B`
samples of the dense directory DIR, binding each argument of `NAME` to the file of DIR with the
argument's name (`.f32` or `.i32`, as S1 binds a `train` declaration), and writes the function's
result for every block, in block order, to FILE. `NAME` need not be `forward` and need not return
a loss: a `fn scores` beside `forward` that shares the model's parameters returns the decider's
candidate scores. Stdout is one line naming the function, the block count, the row count and FILE.
Without `--data`, `infer` behaves as today.

## Context

- Gap row 3. Decider design § 5.3 step 3 (τ per decision type on held-out sessions, split
  conformal prediction, about 400 held-out decisions per type), § 7 (evaluation offline first).
- Today `infer` runs `forward` on the synthetic batch of the `train` block and takes no `--input`
  (`crates/modelspec-cli/src/infer.rs:1-7`; DESIGN.md § 8 and § 12 "Data interface": "`infer
  --input` stay[s] open"). `forward` must return the loss (E-TYPE-017), so no function returns
  scores, and the compiler lowers only the entry the `train` block calls.
- Vulkan: `runner::infer` and `infer_on_device`
  (`crates/modelspec-runtime/src/vulkan/runner.rs:327`, `:487`) run the loss steps once with one
  host wait.
- Codes: E-DATA-009 is the last on main and S1 adds none, so this story takes E-DATA-010.

## Spec first

`spec/domains/data.yaml` gains `modelspec.data.InferenceOutputFile` before any code: the file
`--out` writes, row-major little-endian `f32` (or `i32` for an `i32` result) with no header, one
row per sample of DIR, each row the function's result for that sample, so it reads back with the
layout of `DenseTensorFile`; fields `function`, `dtype`, `row_elements`, `rows`. Then `task
spec-validate`, `task schema-generate`.

## Scope

- `spec/domains/data.yaml`, `crates/modelspec-runtime/schema/`.
- `crates/modelspec-core/src/compile.rs`, `crates/modelspec-core/src/lower.rs`,
  `crates/modelspec-core/src/types.rs` (a second, read-only entry function lowered and checked
  beside `forward`).
- `crates/modelspec-cli/src/infer.rs`, `crates/modelspec-runtime/src/train/data.rs` (the dense
  reader S1 extends, reused), `crates/modelspec-runtime/src/vulkan/runner.rs`.
- `crates/modelspec-cli/tests/infer_data.rs` (new) with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `DESIGN.md` § 8 and § 12 "Data interface"; `crates/modelspec-docs/src/cli.rs` and
  `diagnostics.rs`, then `task docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

`cargo test -p modelspec-cli --test infer_data`:

- **infer-function-cpu**: a fixture with `fn forward` (a loss) and `fn scores` (a `[B, C]`
  linear map sharing `forward`'s weight) is started from known parameters with `--init`, written
  to a checkpoint with `--steps 0 --checkpoint`, and `infer --function scores --data DIR --out
  FILE` on a directory of three blocks writes `3·B·C` values equal, bit for bit, to the product
  the test computes on the host.
- **infer-function-stdout**: in the same run, stdout is exactly one line, naming the function
  (`scores`), the block count (3), the row count (`3·B`) and FILE.
- **infer-without-data-unchanged**: `infer` of the same fixture and checkpoint without `--data`,
  `--function` and `--out` exits 0 and prints the two lines it prints today, `prediction: <type> =
  [...]` and `loss <v>` of `forward` on the synthetic batch of the `train` block, the expected
  text held in the test; and the existing `infer` tests (`checkpoint_cpu.rs`, `checkpoint_vulkan.rs`
  and the `adversary_*` tests that run `infer`) pass with no edit to their expectations.
- **infer-function-refusals**: an unknown `--function` is refused with E-DATA-010 naming the
  model's functions; an argument with no file in DIR with E-DATA-007; a sample count that is not
  a multiple of `B` with E-DATA-009; `--function` or `--out` without `--data` is a usage error
  (exit 2). Nothing is written to FILE on a refusal.
- **infer-function-vulkan** (`#[ignore]`d, `--ignored`): the same on `--backend vulkan`, every
  value within the DESIGN.md § 9 gate 5 tolerance of the CPU file.
- `task ci` exits 0, including `task spec-validate`, `task schema-check` and `task docs-check`.

## Out of scope

- Calibrating τ, conformal prediction and every evaluation metric: the decider repository computes
  them from FILE.
- Packed directories as `--data` for `infer`; streaming output; a function calling another (S5).
- A `fn scores` that reads a `stop_gradient` or `sigreg` term: fixtures here use the 0.21.0
  operators.
