---
format: aep.planning-md/3
id: story:observatory-input-adjoint
kind: story
status: draft
title: 'Input adjoint: gradient of a rank-0 value with respect to a named input'
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: epic:m0-semantic-core
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/tests/input_adjoint.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/input_adjoint.rs
revision: 7
---
# Input adjoint: the gradient of a rank-0 value with respect to a named input

## Context

`epic:training-observatory` asks for gradient x input saliency "since the compiler already derives
gradients" (In scope, Views). It derives them only for loss-reachable trainable parameters
(DESIGN.md § 3.3 pass 4; `differentiate`, `crates/modelspec-core/src/autodiff.rs:50`, whose outputs
are "the loss, then the gradient of each loss-reachable parameter"), never for an `Input`
(`TensorOp::Input`, `crates/modelspec-core/src/ir/tensor.rs:120`). This story adds the adjoint of
a named input. It is a compiler change with no UI. `story:observatory-saliency-gradient-x-input`
uses it; it needs nothing from the observatory stories and builds on the reverse-mode autodiff of
epic:m0-semantic-core (`story:m0-reverse-mode-autodiff`, implemented).

The adjoint lands in a separate program. `TrainingProgram.program` keeps its outputs (the loss, then
one gradient per entry of `gradients`, `crates/modelspec-core/src/ir/training.rs:4-7`), the
`TrainingStep` outputs (then the updates, `training.rs` doc of `TrainingStep`) and what
`modelspec ir --stage training` prints stay as they are. `sgd_step` and the golden files read them.

## What it adds

Two public functions in `crates/modelspec-core/src/autodiff.rs`, with one typed error
`InputAdjointError`:

- `pub fn differentiate_inputs(forward: &TensorProgram, target: ValueId, inputs: &[&str]) ->
  Result<InputAdjoint, InputAdjointError>`. It returns a verified program whose outputs are the
  target, then the adjoint of each named input in the order given, with the input's shape. It
  reuses the gradient-rule registry of DESIGN.md § 4 and adds no operator.
- `pub fn output_element_target(forward: &TensorProgram, prediction: ValueId, index: &[u64]) ->
  Result<OutputTarget, InputAdjointError>`. It builds the rank-0 target for one element of the
  model output from existing ops: it appends an `Input` named `target_mask` with the type of
  `prediction`, then `Mul prediction, target_mask` and `ReduceSum` ("any shape to rank 0",
  `tensor.rs:138`). `OutputTarget` holds the program, the target value and the row-major offset of
  `index`. The caller binds `target_mask` to the one-hot tensor at that offset. The loss target
  needs no builder: it is `CompiledTraining`'s loss value.

Settled here, and written into DESIGN.md § 4: an input the target does not read is not refused. Its
adjoint is an output like any other, all zeros of the input's shape (a `Scalar 0.0` broadcast with
`BroadcastScalar`). Refused with `InputAdjointError`: a name that is no `Input` of the program, a
target that is not rank 0, a `prediction` that is not a value of the program, and an `index` of the
wrong rank or out of bounds. These are library errors with no CLI surface, like `AutodiffError`
(`autodiff.rs:21`), so they get no E- code and `website/docs/reference/diagnostics.md` does not
change.

DESIGN.md: § 3.3 pass 4 and § 4 name the input adjoint and its separate program; the revision line
gets a new newest entry that names it (AGENTS.md Rules: a semantics change edits DESIGN.md and
bumps it).

## Scope

Derived 2026-10-21 by `story-scoper`. Every line is **cited** (read from the story or the tree) or
**inferred** (a reading that could be wrong).

- **Primary surface:** `crates/modelspec-core/src/autodiff.rs` — cited (story What it adds; `differentiate` at :50, `AutodiffError` at :21)
- **Files:** `crates/modelspec-core/tests/input_adjoint.rs` (new) — cited (Acceptance 3, 4)
- **Files:** `crates/modelspec-runtime/tests/input_adjoint.rs` (new) — cited (Acceptance 1, 2)
- **Documents:** `DESIGN.md` § 3.3 pass 4, § 4 and the revision line — cited (Acceptance 5)
- **Symbols:** `differentiate_inputs`, `output_element_target`, `InputAdjoint`, `OutputTarget`, `InputAdjointError` (all new, in `autodiff.rs`) — cited
- **Also likely:** internal reachability helper in `autodiff.rs` (the "depends on a parameter read" filter at :128) gets a from-input variant or becomes generic — inferred
- **Not needed:** `crates/modelspec-core/src/compile.rs` and `lib.rs`. `autodiff` is already `pub mod` (`lib.rs:5`), and `CompiledTraining.forward` exposes `program` and `prediction` as public fields (`compile.rs:37`, `lower.rs:33-36`), so no re-export is required — inferred
- **Not touched:** `CHANGELOG.md`, `Cargo.toml`, `Cargo.lock`, `crates/modelspec-runtime/src/`, `crates/modelspec-docs/`, `website/`, `spec/` — cited (story Scope). No new dependency or operator, so docs-generated pages do not change — inferred
- **Confidence:** high — the story names every file and symbol, and each citation resolves at the stated line
- **Would collide with:** any unit editing `DESIGN.md` (the revision line above all), or `crates/modelspec-core/src/autodiff.rs`
- **Safety fact:** `differentiate` and `TrainingProgram` outputs stay byte-identical because the adjoint is a separate function returning a separate program. The CLI golden `crates/modelspec-cli/tests/golden/linear_regression.training.ir` (`train_linear.rs:22`, test at :235) pins this — step 2, unproven
- **Open for the implementor:** after DCE, `program.inputs` keeps only read inputs (`lower.rs:37-39`), so a declared but unread input of a compiled `.mspec` has no `Input` and is refused rather than zeroed; scenario 2 builds the forward program by hand. No listed refusal covers a program that already has an input named `target_mask`. Whether the per-operand `wanted` mask asks for the `Mse` operand B adjoint (`ops.rs:193`) when `y` is requested is unconfirmed

## Domain relations

None. The input adjoint is compiler-internal; no ESS document types it (AGENTS.md: the `.mspec`
language and its gradient rules are DESIGN.md § 2-4, not an ESS domain).

## Acceptance

Scenarios 1 to 4 are named tests run by `task ci`; scenario 5 is three commands, run from the
repository root, each exiting 0.

1. `observatory.adjoint.finite-difference` (`crates/modelspec-runtime/tests/input_adjoint.rs`): for
   `examples/linear_regression.mspec` and `crates/modelspec-docs/models/two_layer_relu.mspec`, with
   the loss as target and with `output_element_target` at index `[0, 0]`, the adjoint of `x` matches
   central finite differences in `f64` reference math at points away from `relu` kinks, within
   `TOL_F32` = 1e-3 when evaluated in `f32` and `TOL_F64` = 1e-6 when evaluated in `f64`, the
   constants and relative-error measure of `crates/modelspec-runtime/tests/gradients.rs:29-31`.
2. `observatory.adjoint.unread-input`: a forward program with two inputs, the target reading only
   one, asked for both: `differentiate_inputs` succeeds, and on the CPU interpreter the adjoint of
   the unread input is all zeros of its shape, the read one matches its finite difference.
3. `observatory.adjoint.training-unchanged`: `crates/modelspec-cli/tests/train_linear.rs` test
   `ir_stage_training_prints_the_golden` and the tensor golden test pass unchanged, with the golden
   files byte-identical. A new test in `crates/modelspec-core/tests/input_adjoint.rs` asserts that
   the outputs of `compile_training` for both models are, in order, the loss, the gradients, then
   the updates, as before.
4. `observatory.adjoint.refusals` (`crates/modelspec-core/tests/input_adjoint.rs`): each refusal
   named above returns its `InputAdjointError` variant; `output_element_target` for
   two-layer-relu at `[0, 0]` yields a rank-0 target, a `target_mask` input of the prediction's
   type and offset 0; every program either function returns passes the Tensor IR verifier.
5. `observatory.adjoint.design-recorded`:
   - revision line, newest entry: `grep -Eq '^\*\*Revision:\*\* [0-9.]+ \([0-9-]+\): [^;]*input adjoint' DESIGN.md`
   - § 3.3 pass 4: `awk '/^### 3\.3 /{s=1;next} /^##/{s=0} s && /^4\. / && /input adjoint/{f=1} END{exit !f}' DESIGN.md`
   - § 4: `awk '/^## 4\. /{s=1;next} /^## /{s=0} s && /input adjoint/{f=1} END{exit !f}' DESIGN.md`

## Out of scope

- The `Insight` type, its endpoint and the UI (`story:observatory-saliency-gradient-x-input`).
- A CLI surface for the adjoint; `modelspec ir` gains no stage.
