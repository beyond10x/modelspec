---
format: aep.planning-md/3
id: story:observatory-saliency-gradient-x-input
kind: story
status: draft
title: Gradient-times-input saliency for one selected input
summary: For a selected input of two-layer-relu the observatory computes a GradientTimesInput Insight that matches a finite-difference check within 1e-2 and shows it; settles the selected-input and Insight ownership markers.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: story:observatory-ui-live-loss-chart
- depends_on: story:observatory-activation-statistics
- depends_on: story:observatory-input-adjoint
revision: 3
---
# Gradient-times-input saliency for one selected input

## Context

`epic:training-observatory` (In scope, Views: "one explainability method over a selected input
(gradient x input saliency first, since the compiler already derives gradients)"; fourth
Acceptance bullet: saliency for one input of two-layer-relu matches a finite-difference check
within 1e-2). `spec/domains/observatory.yaml` types the result as the entity
`modelspec.observatory.Insight`, method `GradientTimesInput`.

The gradient with respect to an input comes from `story:observatory-input-adjoint`, which adds it
to `modelspec-core` in a separate program and edits DESIGN.md. This story touches neither
`modelspec-core` nor DESIGN.md: it calls `differentiate_inputs`, with the loss or with the target
`output_element_target` builds for an `OutputElement`, multiplies by the input and serves and
draws the result.

## Order on shared files

This story edits `spec/domains/observatory.yaml`, `crates/modelspec-observatory/` and
`ui/src/generated/`, as do `story:observatory-parameter-statistics`,
`story:observatory-architecture-graph` and `story:observatory-activation-statistics`; it is last of
the four and depends on `story:observatory-activation-statistics` for those files. It depends on
`story:observatory-input-adjoint` for the adjoint function.

## Open questions settled first

Two `UNMAPPED:` markers of `spec/domains/observatory.yaml` stand in front of this story, named by
where they sit and a quoted fragment of their text, never by line number. Each is settled before
any code, in that file, with `ess specify validate --path .` in `spec/` passing and the
ESS-ENTITY-019 warning on `modelspec.observatory.Insight` gone. A settlement names its source (the
operator, DESIGN.md or another ESS document); where none answers, the story stops at that step and
reports the question instead of choosing.

1. Marker on `modelspec.observatory.Insight`, text "what a \"selected input\" is": what a selected
   input is (one row of the batch, a whole input tensor, a value the operator supplies) and how the
   UI selects it. Settled with it: which scalar is attributed (the loss, or one element of the
   model output; the epic does not say). It becomes a typed field of `Insight`, for example
   `target`, a union of `Loss` and `OutputElement { index }`, so every insight records what it
   attributes.
2. Marker on `Insight.training_run_id`, text "an insight is \"per step\" of a run": whether the run
   owns its insights or an insight references the run, and whether it is taken against the run or
   against a checkpoint of it. Bound to it: how an insight is computed after the run, from a
   replay, when a recording holds no parameter values (`story:observatory-run-recording-replay`,
   Out of scope).

## Domain relations

- `Insight` to `TrainingRun`: not settled; it is marker 2 above.
- `Insight.input_name` names an input binding by name: the field summary in
  `spec/domains/observatory.yaml` says it is a `DataSpec.name` with role `Input`
  (`spec/domains/artifact.yaml`, type `modelspec.artifact.DataSpec`); no `relations:` entry declares
  it, and M0 has no `CompiledArtifact`, so the name is the Tensor IR `Input` name, inferred from
  `crates/modelspec-core/src/ir/tensor.rs:120` (`TensorOp::Input { name }`).

## Scope

- `crates/modelspec-observatory/`: the `Insight` type with its target field, computing it on the
  CPU interpreter through `differentiate_inputs` and `output_element_target`, its endpoint or event as settled.
- `spec/domains/observatory.yaml` (the two settlements, a `relations:` entry as decided); the
  committed JSON Schema regenerated.
- `ui/`: input selection and a saliency view; `ui/src/generated/` regenerated; `CHANGELOG.md` one
  line, which covers the input adjoint of `story:observatory-input-adjoint` too.
- Not touched: `crates/modelspec-core/`, `DESIGN.md`, `crates/modelspec-runtime/src/`.

## Acceptance

Conformance scenarios, each a named test run by `task ci` except scenario 1, which is the command
shown:

1. `spec/domains/observatory.yaml` no longer contains the markers on `Insight` ("what a \"selected
   input\" is") and on `Insight.training_run_id` ("an insight is \"per step\" of a run"); typed
   declarations that cite their source stand in their place, `Insight` has the typed target field,
   and `ess specify validate --path .` in `spec/` exits 0 without ESS-ENTITY-019 for
   `modelspec.observatory.Insight`.
2. `observatory.insight.saliency-finite-difference`: for
   `crates/modelspec-docs/models/two_layer_relu.mspec` at the last step, every element of the
   `GradientTimesInput` insight for one selected input of `x` is within 1e-2 of input times the
   central finite difference of the scalar the insight's target field names, evaluated in `f64`
   reference math away from `relu` kinks (DESIGN.md § 4, gradient verification). The test runs once
   per target variant the settlement admits.
3. `observatory.insight.schema`: the insight validates against the ESS-generated JSON Schema of
   `modelspec.observatory.Insight`; `values` has as many elements as `shape`.
4. `observatory.ui.saliency-two-layer-relu`: in a headless browser, selecting an input of the
   two-layer-relu run shows a saliency view whose values and target equal the insight the server
   returned.

## Out of scope

- The input adjoint itself and its DESIGN.md edit (`story:observatory-input-adjoint`).
- Other explainability methods (integrated gradients and the like); `InsightMethod` keeps one
  variant.
- Attribution over data files: M0 data is synthetic (DESIGN.md § 8).
