---
format: aep.planning-md/3
id: story:observatory-architecture-graph
kind: story
status: draft
title: 'Architecture view: the diagram graph of the running program, zoomable in place'
summary: The observatory serves the modelspec.diagram graph and renders of the live program and the Vue UI zooms them in place; replaces the observatory ModelGraph and its ownership marker.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: story:observatory-ui-live-loss-chart
- depends_on: story:observatory-parameter-statistics
- depends_on: story:diagram-pipelines
- depends_on: story:diagram-cli
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/observatory_ui.rs
- confidence: cited
  path: crates/modelspec-observatory/Cargo.toml
- confidence: cited
  path: crates/modelspec-observatory/src/server.rs
- confidence: cited
  path: crates/modelspec-observatory/src/typescript.rs
- confidence: cited
  path: ui/src/App.vue
- confidence: inferred
  path: ui/src/components/ArchitectureView.vue
- confidence: inferred
  path: ui/src/generated/diagram.ts
- confidence: inferred
  path: ui/src/lib/diagram.test.ts
- confidence: inferred
  path: ui/src/lib/diagram.ts
revision: 9
---
# Architecture view: the diagram graph of the running program, zoomable in place

## Context

`epic:training-observatory` (In scope, Views; third Acceptance bullet): the model architecture is
drawn from the compiled IR, with operators as nodes carrying their shapes, tensors as edges, and
forward and backward told apart. Revised 2026-10-28: the graph is no longer an observatory noun.
`modelspec.observatory.ModelGraph`, `GraphNode`, `GraphEdge`, `IrStage` and `ModelGraphId` are
replaced by `spec/domains/diagram.yaml` (`modelspec.diagram.DiagramGraph`, `DiagramRender`),
which `epic:ir-svg-gallery` also draws on the documentation site and through `modelspec diagram`
(DESIGN.md § 13). The open question this story used to settle first, the marker on
`ModelGraph.artifact_id`, is settled there: a graph names no artifact and carries the `ir_hash`
that `CompiledArtifact.ir_hash` records for the same source. This story serves that graph from the
training process and draws it in the Vue UI with semantic zoom (DESIGN.md § 13.3).

## What it adds

- `crates/modelspec-observatory/src/server.rs`: `GET /diagram/graph?view=<view>` answers the
  `DiagramGraph` JSON of the program the live `train` runs; `GET /diagram/render?view=&root=&depth=&expand=`
  answers a `DiagramRender` laid out by `modelspec-diagram`; `GET /diagram/glyphs.svg` answers the
  glyph sprite. The request line's query is parsed (today `parse_head` drops it,
  `crates/modelspec-observatory/src/server.rs:319`). An unknown view or scope key answers `400`.
- `crates/modelspec-observatory/Cargo.toml`: depends on `modelspec-diagram` with its `ts` feature.
  The feature and the `ts_rs::TS` derives on the diagram types are already there: the enums of
  `story:diagram-glyph-set` and the `model.rs` types of `story:ir-gallery-renderer`. This story
  edits no file of `crates/modelspec-diagram`.
- `crates/modelspec-observatory/src/typescript.rs` and the `modelspec-ui-types` binary:
  `ui/src/generated/diagram.ts` from those types, written by `task ui-types` and checked by
  `task ui-types-check` beside `observatory.ts`.
- `ui/src/components/ArchitectureView.vue` and `ui/src/lib/diagram.ts`: an Architecture tab with
  `model` and `train` views. Each visible node is a `<g data-key data-op data-shape data-subgraph>`
  with `<use href="/diagram/glyphs.svg#ms-…">`. Pan and zoom are a transform on one group. Clicking
  a composite, or zooming past twice its size, requests the render with it expanded; boxes with the
  same `node_key` move to their new place, and the client computes no layout.

## Order on shared files

This story, `story:observatory-activation-statistics` and `story:observatory-saliency-gradient-x-input`
edit `crates/modelspec-observatory/` and `ui/`. They run in the order architecture-graph,
activation-statistics, saliency. Activation-statistics may key activations by a node's `value_id`.
`spec/domains/observatory.yaml` is not edited by this story. `Cargo.lock` changes with the new
dependency; `story:diagram-cli`, which this story depends on, changes it before.

## Live only in this epic

The graph endpoints serve the program a live `train` is running. A recording holds `RunEvent`s only
(`story:observatory-run-recording-replay`), so a replay has no graph: the server answers the
diagram endpoints with `404` and the UI shows no Architecture tab. Putting the graph into the
recording is not taken here: `RunStarted` is a unit variant whose payload other stories rely on, and
`modelspec diagram <file> --format graph` already draws the same graph offline.

## Scope

- `crates/modelspec-observatory/src/server.rs`, `src/typescript.rs`, `Cargo.toml` (add
  `modelspec-diagram` with `features = ["ts"]`; cited), `Cargo.lock`.
- `crates/modelspec-cli/src/train.rs` (hand the compiled program to the server; cited).
- `ui/src/components/ArchitectureView.vue`, `ui/src/lib/diagram.ts`, `ui/src/lib/diagram.test.ts`
  (new), `ui/src/App.vue` (the tab; cited), `ui/src/generated/diagram.ts` (generated).
- `crates/modelspec-cli/tests/observatory_ui.rs` (the browser cases), `CHANGELOG.md` one line.
- Not `crates/modelspec-diagram/**`.

## Acceptance

Named tests run by `task ci`:

1. `observatory.diagram.graph-matches-ir`: for a live `train --serve` of
   `crates/modelspec-docs/models/two_layer_relu.mspec`, `GET /diagram/graph?view=model` equals
   `modelspec diagram <file> --format graph` byte for byte and validates against the
   `DiagramGraph` JSON Schema.
2. `observatory.diagram.render-matches-cli`: `GET /diagram/render?view=train&depth=0&expand=<Backward key>`
   equals `modelspec diagram <file> --view train --depth 0 --expand <key> --format json`.
3. `observatory.diagram.bad-request`: an unknown view or scope key answers `400`; with no live run
   (`observe` of a recording) every diagram path answers `404`.
4. `observatory.ui.architecture-two-layer-relu` (headless browser, `task ui-browser`): the model
   view has one element per visible node whose `data-op` and `data-shape` equal the render's; in the
   train view, after expanding Forward and Backward, the node elements carry
   `data-subgraph="forward|loss|gradient|optimizer"` matching the graph and all four occur.
5. `observatory.ui.zoom-in-place` (headless browser): clicking the Backward composite replaces it
   with its children without a page navigation; the node elements present before and after with the
   same `data-key` are the same DOM elements.
6. `observatory.ui.replay-no-graph-tab` (headless browser): with the two-layer-relu recording
   replayed and no training process, the UI has no Architecture tab and the loss chart still draws.
7. `observatory.ui-types.diagram`: `task ui-types-check` exits 1 naming `ui/src/generated/diagram.ts`
   while that file is missing or differs by one byte from what `task ui-types` writes, and exits 0
   after `task ui-types`.
8. `diagram.test.ts`: `pnpm test` in `ui/` exits 0 with every case of `ui/src/lib/diagram.test.ts`
   passed, among them one that places the boxes of a two-node render fixture at the render's
   coordinates and one that keeps the element of a `node_key` present in two renders.

## Out of scope

- Kernel IR and the execution DAG (M1, M2).
- Overlaying statistics on graph nodes (`story:observatory-activation-statistics` may link to it).
- The architecture view of a replayed run (see Live only in this epic).
