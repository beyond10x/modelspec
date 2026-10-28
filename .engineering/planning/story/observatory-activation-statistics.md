---
format: aep.planning-md/3
id: story:observatory-activation-statistics
kind: story
status: draft
title: Activation statistics per step
summary: The stream carries typed activation statistics computed from the CPU interpreter's intermediate values and the UI shows them per step; settles the activation statistics marker.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: story:observatory-parameter-statistics
- depends_on: story:observatory-architecture-graph
revision: 4
---
# Activation statistics per step

## Context

`epic:training-observatory` (In scope, Views: "activation statistics"; Outcome: "activation
distributions"). Nothing in `spec/domains/observatory.yaml` types them yet. The CPU interpreter
already returns every intermediate value on request (`evaluate_all`,
`crates/modelspec-runtime/src/cpu/interpreter.rs:85`), so the values are at hand on the CPU
backend; the stream contract must not assume they are free on another backend (epic: nothing in
the stream contract is CPU-specific).

## Order on shared files

This story edits `spec/domains/observatory.yaml`, `crates/modelspec-observatory/` and
`ui/src/generated/`, as do `story:observatory-parameter-statistics`,
`story:observatory-architecture-graph` and `story:observatory-saliency-gradient-x-input`. It
depends on `story:observatory-architecture-graph` for those files and because the settlement may
key activations by its `GraphNode.value_id`. It depends on `story:observatory-parameter-statistics`
for those files, for `runner.rs` and `optimizer.rs`, and for the `Histogram` settled there.

## Open questions settled first

One `UNMAPPED:` marker of `spec/domains/observatory.yaml` stands in front of this story, named by
where it sits and a quoted fragment of its text, never by line number. It is settled before any
code, in that file, with `ess specify validate --path .` in `spec/` passing afterwards. The
settlement names its source (the operator, DESIGN.md or another ESS document); where none answers,
the story stops at that step and reports the question instead of choosing.

1. Marker on `modelspec.observatory.StepCompleted`, text "the epic shows activation statistics per
   step": what activation statistics are keyed by (a Tensor IR value, a layer, a parameter path),
   which values count as activations (every Forward value, or only some ops such as `Relu` and
   `MatMul` outputs), and what each holds (a `Histogram` as settled by
   `story:observatory-parameter-statistics`, a norm, a fraction of zeros after `Relu`). The result
   is a typed field on `StepCompleted` or a new event variant. Whatever it keys by, the set includes
   the `Relu` output of two-layer-relu (scenario 2).

## Domain relations

- Activation statistics to the value they describe: not settled; it is the marker above. If they
  are keyed by Tensor IR value, the key is a `GraphNode.value_id` of the run's Tensor-stage graph
  (`spec/domains/observatory.yaml`, type `modelspec.observatory.GraphNode`), and the settlement says
  so in a field summary.

## Scope

- `crates/modelspec-runtime/src/train/runner.rs`, `optimizer.rs`: the observer receives the
  step's intermediate values (through `evaluate_all`; the interpreter is not edited).
- `crates/modelspec-observatory/`: activation statistics computation and the event field.
- `spec/domains/observatory.yaml` (the settlement); the committed JSON Schema regenerated.
- `ui/`: an activation view per step; `ui/src/generated/` regenerated; `CHANGELOG.md` one line.

## Acceptance

Conformance scenarios, each a named test run by `task ci` except scenario 1, which is the command
shown:

1. `spec/domains/observatory.yaml` no longer contains the marker on `StepCompleted` ("the epic
   shows activation statistics per step"); a typed declaration that cites its source stands in its
   place, and `ess specify validate --path .` in `spec/` exits 0.
2. `observatory.activations.two-layer-relu`: for `crates/modelspec-docs/models/two_layer_relu.mspec`,
   the activation statistics of the event at step `k` equal those the test computes from
   `evaluate_all` of the forward program on that step's inputs and the parameters after `k`
   updates, for every activation the settlement names, which includes at
   least the `Relu` output of two-layer-relu; every event validates against the
   ESS-generated JSON Schema.
3. `observatory.ui.activations-two-layer-relu`: in a headless browser, the activation view of
   two-layer-relu shows, for the selected step, the streamed statistics of each activation.

## Out of scope

- Attributions (`story:observatory-saliency-gradient-x-input`).
- Activation statistics on any backend other than CPU.

## Graph types moved (2026-10-28)

On 2026-10-28 `modelspec.observatory.GraphNode` and the observatory's other graph types were replaced
by `spec/domains/diagram.yaml` (DESIGN.md § 13, `story:observatory-architecture-graph` revision 5).
Where this story speaks of a `GraphNode.value_id` of the run's Tensor-stage graph, read the
`value_id` of an Operator `modelspec.diagram.DiagramNode` in the run's `model` or `train`
`DiagramGraph`; it is the same Tensor IR `ValueId`.
