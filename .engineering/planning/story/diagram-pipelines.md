---
format: aep.planning-md/3
id: story:diagram-pipelines
kind: story
status: draft
title: 'Pipeline views: train, eval and infer, each stage a composite that zooms into the model'
summary: Train, eval and infer graphs whose root children are pipeline stages placed from IR tags and scopes, with endpoints, adjoint and update composites, readable at BERT-tiny size.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:diagram-zoom
scope:
- confidence: inferred
  path: crates/modelspec-diagram/src/graph.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/lib.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/pipeline.rs
- confidence: inferred
  path: crates/modelspec-diagram/tests/pipelines.rs
revision: 2
---
# Pipeline views: train, eval and infer, each stage a composite that zooms into the model

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.4. Above the model view stand the pipelines: the training
step (data, parameters, forward, loss, backward, optimizer update, checkpoint), the evaluation and
inference. The Training IR already tags every node `forward`, `loss`, `gradient` or `optimizer`
(`Subgraph`, `crates/modelspec-core/src/ir/training.rs`; `TrainingProgram.subgraphs`), and
`modelspec ir --stage training` prints the tag on each line. The stages are those tags plus the
IR's inputs and parameter state. They are not a second reading of the source.

## What it adds

- `crates/modelspec-diagram/src/pipeline.rs`: the `train`, `eval` and `infer` graphs. Root children
  are the `PipelineStage` composites of DESIGN.md § 13.4; Operator nodes are placed by subgraph (and
  `Input` and `ParamRead` by role). Endpoints: DataSource (Train split in `train`, HeldOut in `eval`),
  CheckpointSink (`train`, the parameters at version `n + 1`), CheckpointSource and PredictionSink
  (`infer`). `eval` and `infer` read the program `lower::without_dropout` makes.
- Under each stage the IR scopes: Backward holds one `scope-adjoint` composite per forward node with
  a gradient rule, in the function scope of that node; Optimizer one `scope-update-<optimizer>`
  composite per updated parameter.
- Default depth 0 for the pipeline views and 1 for the model view.

## Scope

- `crates/modelspec-diagram/src/pipeline.rs`, `tests/pipelines.rs` (new); `src/graph.rs` and
  `src/lib.rs` (cited from earlier stories of the epic).
- Not `crates/modelspec-core/**`.

## Acceptance

Tests in `crates/modelspec-diagram/tests/pipelines.rs` (`cargo test -p modelspec-diagram`, `task ci`):

1. `pipeline.train-stages`: the `train` render of `examples/linear_regression.mspec` at depth 0 has
   the composites Data, Parameters, Forward, Loss, Backward, Optimizer and the CheckpointSink and
   DataSource endpoints, with edges Data→Forward, Parameters→Forward, Forward→Loss, Loss→Backward,
   Backward→Optimizer, Parameters→Optimizer, Optimizer→CheckpointSink, and no edge into Parameters.
2. `pipeline.stage-membership`: every Operator node of the `train` graph of
   `two_layer_relu.mspec` lies in the stage the placement rule of DESIGN.md § 13.4 gives it, and
   keeps in `subgraph` the tag `ir --stage training` prints for it; the `forward`-tagged rule values
   (`%9 Sub`, `%10 Scalar`, `%13`, `%15` and `%18 Transpose` in today's print) lie in Backward.
3. `pipeline.forward-is-model`: the Operator nodes under Forward of the `train` graph are exactly
   the `model` graph's Operator nodes other than `Input` and `ParamRead`; in a fixture whose `loss`
   applies an operator outside the `model.forward(...)` call (such as `mul(mse(model.forward(x), y),
   0.5)`), that operator's nodes lie in Loss, not Forward.
4. `pipeline.adjoints-and-updates`: for `examples/mnist_mlp.mspec` the Backward stage holds one
   Adjoint composite per forward node with a gradient rule, and Optimizer one Update composite per
   parameter.
5. `pipeline.eval-infer`: `eval` and `infer` of a file with `dropout` have no `Dropout` node and no
   Backward or Optimizer stage; `infer` has the CheckpointSource and PredictionSink endpoints.
6. `pipeline.bert-tiny-readable`: every level of the `model` and `train` graphs of
   `examples/bert_tiny.mspec`, drawn at its default depth and with each single scope expanded, has
   at most 80 visible nodes, no overlapping boxes and no route through a box (DESIGN.md § 13.5).
7. `pipeline.acyclic`: every render of every view of the files `task check-examples` checks is
   acyclic.

## Out of scope

- `infer --function` (`story:diagram-infer-function-view`).
- The command line (`story:diagram-cli`) and pages (`story:diagram-docs-blocks`,
  `story:ir-gallery-modules`).
