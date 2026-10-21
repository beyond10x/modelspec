---
format: aep.planning-md/3
id: story:observatory-architecture-graph
kind: story
status: draft
title: 'Architecture view: the ModelGraph of the Tensor IR and Training IR'
summary: The UI draws the two-layer-relu ModelGraph derived from verified Tensor and Training IR, every operator with its shape and forward/backward subgraphs told apart; settles the ModelGraph ownership marker.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: story:observatory-ui-live-loss-chart
- depends_on: story:observatory-parameter-statistics
revision: 3
---
# Architecture view: the ModelGraph of the Tensor IR and the training step

## Context

`epic:training-observatory` (In scope, Views; third Acceptance bullet): the model architecture is
drawn from the compiled IR, nodes are operators with shapes, edges are tensors, forward and
backward told apart. `spec/domains/observatory.yaml` types the result as the entity
`modelspec.observatory.ModelGraph` (`GraphNode`, `GraphEdge`, `IrStage`, `Subgraph`,
`TensorOperator`). This story derives it in Rust from the same verified IR that
`modelspec ir --stage tensor|training` prints, serves it to the UI and draws it.

The Training stage is derived from `TrainingStep` (`CompiledTraining.training`,
`crates/modelspec-core/src/compile.rs`), not from `TrainingProgram`: `attach_sgd` adds the SGD nodes
and tags them `Subgraph::Optimizer` in `training.subgraphs`
(`crates/modelspec-core/src/ir/training.rs:192`, `:225`), so only the step has Optimizer nodes. It is
what `ir --stage training` prints (`crates/modelspec-cli/src/ir.rs:4`; golden
`crates/modelspec-cli/tests/golden/linear_regression.training.ir` lines 15-17 are `optimizer`).

## Order on shared files

This story, `story:observatory-parameter-statistics`, `story:observatory-activation-statistics` and
`story:observatory-saliency-gradient-x-input` all edit `spec/domains/observatory.yaml`,
`crates/modelspec-observatory/` and `ui/src/generated/`. They run in the order parameter-statistics,
architecture-graph, activation-statistics, saliency. This story depends on
`story:observatory-parameter-statistics` for those shared files. Activation-statistics comes after
it because its settlement may key activations by `GraphNode.value_id` of this graph.

## Live only in this epic

The graph endpoint serves the graph of the program a live `train` is running. A recording holds
`RunEvent`s only (`story:observatory-run-recording-replay`), so a replay has no graph: the server
answers the graph endpoint with 404 and the UI shows no architecture tab. The epic says the UI also
works after the run (epic, In scope); for the architecture view that is out of scope here and
recorded as such. The cheap way to put the graph into a recording, a `graph` field on `RunStarted`,
is not taken: `RunStarted` is a unit variant whose payload the live story settles (marker on
`RunEventBody`, "RunStarted, RunFinished and RunFailed are lifecycle changes of the run"), and the
field would re-record both committed fixtures and change a contract three earlier stories built on.

## Open questions settled first

One `UNMAPPED:` marker of `spec/domains/observatory.yaml` stands in front of this story, named by
where it sits and a quoted fragment of its text, never by line number. It is settled before any
code, in that file, with `ess specify validate --path .` in `spec/` passing and its ESS-ENTITY-019
warning on `modelspec.observatory.ModelGraph` gone. The settlement names its source (the operator,
DESIGN.md or another ESS document); where none answers, the story stops at that step and reports
the question instead of choosing.

1. Marker on `ModelGraph.artifact_id`, text "a graph is derived from one artifact's IR": whether
   `CompiledArtifact` owns its graphs or a graph references its artifact, and whether an artifact
   has one graph per `IrStage`. Bound to it, because the answer needs it: what
   `ModelGraph.artifact_id` names under M0, which writes no `CompiledArtifact`
   (`story:m0-cpu-sgd-training`, Out of scope).

## Domain relations

- `ModelGraph` to `CompiledArtifact`: not settled; it is the marker above.
- `GraphNode` from Tensor IR `TensorNode`: one node per SSA definition, `value_id` the
  `ValueId`, `op` the `TensorOp` variant, dtype and shape the node type, `subgraph` the tag in
  `TrainingStep.training.subgraphs`. Inferred from `crates/modelspec-core/src/ir/tensor.rs:175`
  (`TensorNode { id, op, operands, ty }`) and `crates/modelspec-core/src/ir/training.rs:66`
  (`subgraphs`); no ess/1 relation declares the mapping, `GraphNode` field summaries in
  `spec/domains/observatory.yaml` describe it.
- `GraphEdge` from an operand: one edge per entry of `TensorNode.operands`, `operand` its
  position. Inferred from `crates/modelspec-core/src/ir/tensor.rs:175`; no ess/1 relation declares
  it.

## Scope

- `crates/modelspec-observatory/`: `ModelGraph` derivation from `TensorProgram` (Tensor stage) and
  `TrainingStep` (Training stage); it reads `modelspec-core`, which stays free of Vulkan and of the
  observatory. The graph endpoint; the generated TypeScript for the graph types.
- `spec/domains/observatory.yaml` (the settlement, and a `relations:` entry if one is decided).
- `ui/`: an architecture view, Tensor and Training stages, each node carrying `data-subgraph`,
  `data-op` and `data-shape`.
- `ui/src/generated/` regenerated; `CHANGELOG.md` one line.
- Tests: a Rust test of the derivation and the headless browser test of the view.

## Acceptance

Conformance scenarios, each a named test run by `task ci` except scenario 1, which is the command
shown:

1. `spec/domains/observatory.yaml` no longer contains the marker on `ModelGraph.artifact_id` ("a
   graph is derived from one artifact's IR"); a typed declaration that cites its source stands in
   its place, and `ess specify validate --path .` in `spec/` exits 0 without ESS-ENTITY-019 for
   `modelspec.observatory.ModelGraph`.
2. `observatory.graph.tensor-matches-ir`: for
   `crates/modelspec-docs/models/two_layer_relu.mspec`, the Tensor-stage graph has exactly one node
   per definition `modelspec ir --stage tensor` prints, each with that line's op, dtype and shape,
   and one edge per operand; it validates against the ESS-generated JSON Schema of
   `modelspec.observatory.ModelGraph`.
3. `observatory.graph.training-subgraphs`: the Training-stage graph of the same file has one node
   per definition `modelspec ir --stage training` prints, tagged with the subgraph that line starts
   with (`forward`, `loss`, `gradient`, `optimizer`); it has at least one node of each.
4. `observatory.ui.architecture-two-layer-relu`: in a headless browser, the Tensor stage of the
   architecture view of two-layer-relu has one element per Tensor IR definition whose `data-op` and
   `data-shape` equal the printed op and shape; in the Training stage each node element has
   `data-subgraph="forward|loss|gradient|optimizer"` equal to the tag scenario 3 checks, and the
   four values occur.
5. `observatory.ui.replay-no-graph-tab`: in a headless browser, with the two-layer-relu fixture
   recording replayed and no training process running, the UI has no architecture tab and the
   graph endpoint answers 404; the loss chart still draws.

## Out of scope

- Kernel IR and the execution DAG (M1, M2).
- Overlaying statistics on graph nodes (`story:observatory-activation-statistics` may link to it).
- The architecture view of a replayed run (see Live only in this epic).
