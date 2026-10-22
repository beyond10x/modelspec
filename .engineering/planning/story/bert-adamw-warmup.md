---
format: aep.planning-md/3
id: story:bert-adamw-warmup
kind: story
status: draft
title: AdamW with decoupled weight decay and a warmup schedule, compiled into the training step
summary: Adds adamw(...) as a compiled Optimizer subgraph with versioned moment state and a host-computed schedule; settles the LearningRateSchedule and OptimizerConfig.schedule markers.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-dropout
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_adamw.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/
- confidence: cited
  path: crates/modelspec-cli/tests/train_adamw.rs
- confidence: cited
  path: crates/modelspec-core/src/compile.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/training.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/ast.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-observatory/
- confidence: cited
  path: crates/modelspec-observatory/schema/
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/optimizer.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adamw.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: spec/domains/training.yaml
- confidence: cited
  path: ui/src/generated/
- confidence: cited
  path: website/docs/reference/
revision: 3
---
# AdamW with decoupled weight decay and a warmup schedule, compiled into the training step

## Context

`epic:bert-from-scratch` (In scope, Language and operators: "AdamW with weight decay and warmup").
M0 has SGD only: the parser refuses any other optimizer (`crates/modelspec-core/src/syntax/parser.rs`,
`fn optimizer`, "only `sgd` is"), and `attach_sgd` (`crates/modelspec-core/src/ir/training.rs:192`)
appends the update to the training step as its `Optimizer` subgraph, with a `Scalar(lr)` node.
DESIGN.md § 0 keeps "training is compiled", so the AdamW update is IR too, not host code.
`spec/domains/training.yaml` already lists `AdamW` in `OptimizerKind` and its fields in
`OptimizerConfig`, and says "The CLI accepts only `sgd` until then".

## Open questions settled first

Two `UNMAPPED:` markers of `spec/domains/training.yaml` stand in front of this story, named by
where they sit and a quoted fragment of their text, never by line number. Each is settled before
any code, in that file, with `ess specify validate --path .` in `spec/` passing afterwards. A
settlement names its source (the operator, DESIGN.md or another ESS document); where none answers,
the story stops at that step and reports the question instead of choosing.

1. Marker on `modelspec.training.LearningRateSchedule`, text "the epic names warmup only": the
   rate after warmup (constant, linear decay to zero, another shape) and whether warmup starts at
   zero. For the operator: the BERT paper (Devlin et al. 2019) warms up linearly from 0 and then
   decays linearly to 0 at the last step; that is a candidate, not a source.
2. Marker on `OptimizerConfig.schedule`, text "whether a schedule applies to SGD too or only to
   AdamW": settled as AdamW only. Source: the epic names the schedule only with AdamW (In scope,
   Language and operators: "AdamW with weight decay and warmup"), and nothing in the epic or
   DESIGN.md asks for a schedule on `sgd`. `sgd(lr = F)` keeps its M0 form and a constant rate. If
   the operator answers otherwise before the story starts, this body is revised first.

## What this story decides

- Syntax: `optimizer = adamw(lr = F, weight_decay = F, beta1 = F, beta2 = F, eps = F, ...)`, all
  named and required, followed by the schedule arguments the first settlement defines. A value out
  of range (`lr <= 0`, `weight_decay < 0`, a `beta` outside `[0, 1)`, `eps <= 0`) is refused with a
  code at the argument.
- Update, for every parameter with a gradient, at step `t` counted from 1:
  `m = beta1 m + (1 - beta1) g`, `v = beta2 v + (1 - beta2) g^2`,
  `p = p - lr_t (m / (1 - beta1^t) / (sqrt(v / (1 - beta2^t)) + eps) + weight_decay p)`
  (Loshchilov and Hutter, decoupled weight decay, as the summary of `OptimizerConfig` names).
  `lr_t` and the two bias corrections are rank-0 inputs of the step that the training host
  computes in `f64` each step from the settled schedule; everything else is in the `Optimizer`
  subgraph, with internal elementwise ops `Sqrt` and `Div`. Weight decay applies to every
  parameter: the language has no parameter groups.
- `m` and `v` are optimizer state of each parameter, versioned and threaded through effect tokens
  as parameters are (`attach_sgd` doc), starting at 0.

## Order on shared files

Last of the seven operator stories; the shared files are listed in `story:bert-embedding-lookup`
(Order on shared files). This story depends on `story:bert-dropout`, which edits
`crates/modelspec-core/src/ir/training.rs`, `crates/modelspec-runtime/src/train/runner.rs` and
`optimizer.rs` before it. `story:bert-mlm-batches` depends on this story for `DESIGN.md` and
`spec/domains/training.yaml`.

## Domain relations

- `modelspec.training.TrainingRun.optimizer` is a field of type
  `modelspec.training.OptimizerConfig` (`spec/domains/training.yaml`), so the run records the
  AdamW hyperparameters and the schedule; the invariant there ties the AdamW fields to
  `kind == AdamW`.
- `modelspec.checkpoint.Checkpoint.optimizer_slots` (`spec/domains/checkpoint.yaml`, "empty for
  plain SGD") is where `m` and `v` go once checkpoints exist; this story writes no checkpoint
  (DESIGN.md § 6.2 is M1).

## Scope

- `spec/domains/training.yaml` (the two settlements; the `OptimizerKind` summary no longer says only
  `sgd` is accepted).
- `crates/modelspec-core/src/syntax/{parser,ast}.rs`, `types.rs` (hyperparameter checks and codes),
  `ir/training.rs` (`attach_adamw`), `ir/tensor.rs` (`Sqrt`, `Div`), `ir/verify.rs` (`verify_step`
  for optimizer state), `ir/print.rs`, `compile.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs` (`Sqrt`, `Div`),
  `src/train/optimizer.rs` (the AdamW step, moment state), `runner.rs` (the per-step scalars).
- `crates/modelspec-runtime/tests/adamw.rs` (new); `crates/modelspec-cli/tests/train_adamw.rs`
  and `check_adamw.rs` (new) with fixtures under `crates/modelspec-cli/tests/fixtures/`.
- `spec/domains/observatory.yaml` (`TensorOperator`), the committed observatory JSON Schema,
  `crates/modelspec-observatory/`, `ui/src/generated/`.
- `DESIGN.md` § 2.1 (the optimizer setting), § 6 (optimizer state), § 8 and the revision line;
  `website/docs/reference/` through `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

A `train` block with `adamw(...)` and a warmup schedule trains on the CPU reference, each step
equal to the AdamW update computed in `f64` by the test, with the learning rate the settled
schedule gives.

Scenarios, each run by `task ci` through the command shown, except scenario 1:

1. `spec/domains/training.yaml` no longer contains the markers on `LearningRateSchedule` ("the
   epic names warmup only") and on `OptimizerConfig.schedule` ("whether a schedule applies to SGD
   too"); typed declarations that cite their source stand in their place, and
   `cd spec && ess specify validate --path .` exits 0.
2. `cargo test -p modelspec-runtime --test adamw one_step`: for a fixture model with two
   parameters, the parameters after steps 1, 2 and 3 equal those the test computes from the
   update above in `f64` within `1e-6` relative, with `m` and `v` starting at 0.
3. `cargo test -p modelspec-runtime --test adamw schedule`: the `lr_t` the runner passes at every
   step from 0 to `warmup_steps + 2` equals the settled schedule evaluated in `f64`.
4. `cargo test -p modelspec-runtime --test adamw decoupled_decay`: for a parameter whose gradient is
   exactly 0 (read only through `relu` of negative values), one step multiplies it by
   `1 - lr_t weight_decay`.
5. `cargo test -p modelspec-cli --test train_adamw`: `modelspec train --backend cpu` on the linear
   regression model with `adamw(...)` exits 0, its last printed loss is below its step-0 loss, and
   two runs with one seed print identical stdout; `modelspec ir --stage training` on it prints an
   `Optimizer` subgraph with `Sqrt` and `Div` nodes.
6. `cargo test -p modelspec-cli --test check_adamw`: a missing `beta2`, `beta1 = 1.0`, `eps = 0.0`
   and `weight_decay = -0.1` each exit 1 with a code at the argument; `sgd(lr = 0.1, ...)` with a
   schedule argument of the first settlement (for example `warmup_steps = 10`) exits 1 with
   `E-SYNTAX-002` at that argument, as M0 refuses any argument after `lr` today
   (`crates/modelspec-core/src/syntax/parser.rs`, `fn optimizer`, which expects `)` after the
   `lr` value); and `cargo test -p modelspec-cli --test train_linear` passes unchanged, so `sgd`
   keeps a constant rate.
7. `task docs-check` exits 0.

## Out of scope

- AdamW moments in checkpoints and resume (M1 checkpoints; not in the epic).
- Parameter groups, such as the exclusion of bias and layer-norm weights from decay in BERT.
- The AdamW kernel on the GPU (`story:bert-vulkan-kernels-attention-loss-adamw`).
