---
format: aep.planning-md/3
id: story:decider-const-tensors
kind: story
status: draft
title: const tensors with index generators
summary: Read-only shape-only tensors from arange, repeat_each, tile and fill, computed at compile time and uploaded once.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
- depends_on: story:decider-loss-terms-and-rank-probe
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/check_const.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/train_const.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/hir.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/ast.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/mod.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-docs/src/grammar.rs
- confidence: cited
  path: crates/modelspec-observatory/schema
- confidence: cited
  path: crates/modelspec-runtime/schema
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/stepwise.rs
- confidence: cited
  path: spec/domains/artifact.yaml
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 5
---
# const tensors with index generators

`depends_on: story:decider-loss-terms-and-rank-probe`; lands sixth of the epic's eight, see the
epic's § Landing order.

## Outcome

A model may declare read-only tensors whose values depend only on their shape, from a closed set of
generators: `const evt_row: Tensor<i32, [256]> = arange(0, 256);` (start, step), `repeat_each(c,
k)` (each element `k` times), `tile(c, k)` (the whole `k` times) and `fill(v)`, for `f32` and
`i32`. A `const` is never trained, never in a checkpoint, computed when the program is compiled
and uploaded once per run, and reads like any other tensor in a model function.

## Context

- Gap row 4: 7 of the decider's 19 per-step tensors (`evt_row`, `last_row`, `pair_ctx`,
  `pair_act`, `fold_rows`, `fold_cols`, `fold_bcast`) depend only on the shapes, and ship as data
  every step today (`fold_rows` alone is 4.3 MB a step). `const` is E-SYNTAX-010 ("arrives with
  DESIGN.md § 11 iteration 3", `crates/modelspec-core/src/syntax/mod.rs:53-54`); DESIGN.md § 2.1
  lowers `const` to a "Constant IR value".
- Once the `reshape` of `story:decider-shape-operators` lands, the three `fold_*` tensors go away;
  the remaining four need `arange`, `repeat_each` and `tile`. The name `repeat` is taken:
  E-SYNTAX-011 reserves the `repeat` construct (`syntax/mod.rs:55-56`).
- Codes: the stories that land before this one take E-TYPE-027 to E-TYPE-034 (E-TYPE-026 is
  `story:jepa-stop-gradient`'s), so this story takes E-TYPE-035 to E-TYPE-037.

## Spec first

Before any code: `spec/domains/artifact.yaml` gains `modelspec.artifact.ConstSpec` (name, dtype,
shape, the generator and its arguments, the buffer view the runner uploads it to) and the
compiled artifact a list of them; `spec/domains/observatory.yaml` `TensorOperator` gains `Const`.
Then `task spec-validate`, `task schema-generate` (both `schema/` trees), `task ui-types`.

## Scope

- `spec/domains/artifact.yaml`, `spec/domains/observatory.yaml`,
  `crates/modelspec-observatory/schema/`, `crates/modelspec-runtime/schema/`, `ui/src/generated/`.
- `crates/modelspec-core/src/`: `syntax/parser.rs`, `syntax/ast.rs`, `syntax/mod.rs`
  (E-SYNTAX-010 retired), `resolve.rs`, `types.rs`, `hir.rs`, `lower.rs`, `ir/tensor.rs`,
  `ir/print.rs`, `artifact.rs`, `schedule.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`, `crates/modelspec-runtime/src/vulkan/runner.rs`,
  `crates/modelspec-runtime/src/vulkan/stepwise.rs`.
- Tests (new): `crates/modelspec-cli/tests/check_const.rs`, `crates/modelspec-cli/tests/train_const.rs`,
  fixtures under `crates/modelspec-cli/tests/fixtures/`.
- `DESIGN.md` § 2.1, § 2.3, § 8.1; `crates/modelspec-docs/src/grammar.rs` and `diagnostics.rs`,
  then `task docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

- **const-generators**: `cargo test -p modelspec-cli --test check_const`: `ir --stage tensor` of a
  fixture prints the values of `arange(0, 4)`, `arange(3, 256)`, `repeat_each(arange(0, 3), 2)`,
  `tile(arange(0, 3), 2)` and `fill(1.0)` as the test expects; each refused at the declaration or
  use: a declared shape whose element count the generator cannot fill with E-TYPE-035, a `const`
  used as a parameter with E-TYPE-036, and an `f32` generator for an `i32` const with E-TYPE-037.
- **const-equals-data-cpu**: `cargo test -p modelspec-cli --test train_const`: a fixture that
  multiplies by an `f32` const, and the same program reading those values from a dense `--data`
  `.f32` file, print identical `step` lines over 10 CPU steps.
- **const-row-ids-cpu**: in the same test, a fixture reading an `i32` const as `embedding` row ids
  prints the step-0 loss the test computes on the host from the initial parameters.
- **const-not-in-checkpoint**: a checkpoint of the fixture holds no record for a const.
- **const-vulkan** (`#[ignore]`d, `--ignored`): the fixture on `--backend vulkan` within the
  DESIGN.md § 9 gate 5 tolerance of the CPU run, with the const uploaded once (`--stats`).
- `task ci` exits 0, including `task spec-validate`, `task schema-check`, `task ui-types-check`
  and `task docs-check`.

## Out of scope

- A block one-hot generator: the `reshape` of `story:decider-shape-operators` removes the
  `fold_*` tensors that needed it.
- Constants computed from other expressions, constants read from files, `repeat` (E-SYNTAX-011).
- Rewriting the decider fixture: the epic's closing check.
