---
format: aep.planning-md/3
id: story:observatory-parameter-statistics
kind: story
status: draft
title: 'Parameter and gradient statistics per step: norms and histograms'
summary: StepCompleted carries per-parameter value and gradient norms and histograms from the CPU step, drawn in the UI; settles the histogram, norm and sampling markers.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: story:observatory-ui-live-loss-chart
revision: 4
---
# Parameter and gradient statistics per step: norms and histograms

## Context

`epic:training-observatory` (In scope, Views: "parameter/gradient histograms and norms per step").
`StepCompleted.parameters` (`spec/domains/observatory.yaml`) carries one `ParameterStatistics` per
parameter and `StatisticsSubject` (`Value`, `Gradient`). `story:observatory-live-event-stream`
declares these Rust types, with `Histogram`, and sends the list empty. This story fills it from the
CPU training step and draws it.

The values exist already: `sgd_step` evaluates the training program, whose outputs are the loss
then each gradient in parameter order (`crates/modelspec-core/src/ir/training.rs:4-6`), and keeps
only the loss (`crates/modelspec-runtime/src/train/optimizer.rs`). The step returns its gradients
too; `src/cpu/interpreter.rs` is not edited.

## Order on shared files

This story is the first of four that edit `spec/domains/observatory.yaml`,
`crates/modelspec-observatory/` and `ui/src/generated/`; the order is parameter-statistics, then
`story:observatory-architecture-graph`, then `story:observatory-activation-statistics`, then
`story:observatory-saliency-gradient-x-input`, each with a `depends_on` edge to the one before.

## Open questions settled first

Three `UNMAPPED:` markers of `spec/domains/observatory.yaml` stand in front of this story, named by
where they sit and a quoted fragment of their text, never by line number. Each is settled before
any code, in that file, with `ess specify validate --path .` in `spec/` passing afterwards. A
settlement names its source (the operator, DESIGN.md or another ESS document); where none answers,
the story stops at that step and reports the question instead of choosing.

1. Marker on `modelspec.observatory.Histogram`, text "the bin count and how the edges are chosen":
   the bin count and how edges are chosen (fixed, per step, per tensor).
   `story:observatory-activation-statistics` reuses the answer.
2. Marker on `ParameterStatistics.norm`, text "the epic says \"norms\" and not which norm": which
   norm (L2, max).
3. Marker on `StepCompleted.parameters`, text "whether every step carries statistics or only
   sampled steps": every step, or sampled steps with the list empty otherwise, and if sampled, how
   the interval is set.

## What the post-loop event carries

The runner records `steps + 1` losses; the last, at `k = steps`, is the forward program on the final
parameters with no `sgd_step` and so no gradient (`runner.rs` module doc and `run`). That event
carries statistics whatever the sampling: one `Value` entry per parameter and no `Gradient` entry.

## Domain relations

- A `ParameterStatistics` names its parameter by `path`, typed
  `modelspec.checkpoint.ParameterPath` (`spec/domains/observatory.yaml`); no `relations:` entry
  links it to a parameter declaration. The path is the canonical `ParamDecl.path`, inferred from
  `crates/modelspec-core/src/ir/tensor.rs:199`.
- A parameter the loss does not read has no gradient, so it has a `Value` entry and no
  `Gradient` entry: `StepCompleted.parameters` summary in `spec/domains/observatory.yaml`, inferred
  from `crates/modelspec-core/src/ir/training.rs:73` (`gradients` holds only loss-reachable
  parameters).

## Scope

- `crates/modelspec-runtime/src/train/optimizer.rs`, `runner.rs`: the step returns its gradients;
  the per-step observer receives values and gradients.
- `crates/modelspec-observatory/`: norm and histogram computation, `StepCompleted.parameters`.
- `spec/domains/observatory.yaml` (the three settlements); the committed JSON Schema regenerated.
- `ui/`: per-parameter histogram for the selected step and norm-over-steps chart, value and
  gradient; `ui/src/generated/` regenerated; `CHANGELOG.md` one line.

## Acceptance

Conformance scenarios, each a named test run by `task ci` except scenario 1, which is the command
shown:

1. `spec/domains/observatory.yaml` no longer contains the markers on `Histogram` ("the bin count and
   how the edges are chosen"), on `ParameterStatistics.norm` ("not which norm") and on
   `StepCompleted.parameters` ("whether every step carries statistics"); typed declarations that
   cite their source stand in their place, and `ess specify validate --path .` in `spec/` exits 0.
2. `observatory.stats.values-at-step`: for `examples/linear_regression.mspec`, in every event
   `StepCompleted { step: k }` that carries statistics, the `Value` statistics of `LinearReg.w`
   equal the norm and histogram the test computes from the parameter values after `k` updates
   (`state.values` of `runner::run` with `steps = k`, as `runner.rs` test
   `the_loss_of_step_k_is_the_loss_after_k_updates` relies on). At least one event carries
   statistics, and the event for `k = steps` carries `Value` entries, whatever sampling the
   settlement chose.
3. `observatory.stats.gradient-at-step`: in every such event with `k < steps`, the `Gradient`
   statistics equal those of the gradient a direct `sgd_step` on the parameters after `k` updates
   produces; the event at `k = steps` has `Value` entries only (at least one event with `k < steps`
   carries statistics when `steps > 0`);
   histogram counts sum to the element count; every event validates against the ESS-generated
   JSON Schema.
4. `observatory.stats.unread-parameter`: a model with a parameter the loss does not read (as in
   `runner.rs` test `a_parameter_the_loss_does_not_read_keeps_its_value_and_version`) streams a
   `Value` entry and no `Gradient` entry for it.
5. `observatory.ui.parameter-stats-two-layer-relu`: in a headless browser, for two-layer-relu, the
   norm chart of `TwoLayer.w1` has its last point equal to the last streamed norm, and the histogram
   of the selected step has the streamed bin counts.

## Out of scope

- Activation statistics (`story:observatory-activation-statistics`).
- Optimizer-state statistics (AdamW moments, M3).
