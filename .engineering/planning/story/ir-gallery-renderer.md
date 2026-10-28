---
format: aep.planning-md/3
id: story:ir-gallery-renderer
kind: story
status: draft
title: 'Diagram renderer: DiagramGraph from verified IR, layered layout, SVG and JSON'
summary: modelspec-diagram builds the DiagramGraph of the model view from verified IR, lays it out deterministically and writes SVG with glyphs and schema-valid JSON.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:diagram-glyph-set
scope:
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: Taskfile.yml
- confidence: inferred
  path: crates/modelspec-diagram/Cargo.toml
- confidence: inferred
  path: crates/modelspec-diagram/schema
- confidence: inferred
  path: crates/modelspec-diagram/src/graph.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/json.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/layout.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/lib.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/model.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/svg.rs
- confidence: inferred
  path: crates/modelspec-diagram/tests/render.rs
revision: 7
---
# Diagram renderer: DiagramGraph from verified IR, layered layout, SVG and JSON

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.2 and § 13.5. Revised 2026-10-28: the renderer now lives in
`crates/modelspec-diagram` (created by `story:diagram-glyph-set`), not in `modelspec-docs`, because
three crates draw with it (docs, CLI, observatory; DESIGN.md § 13.8). It builds the
`modelspec.diagram.DiagramGraph` of `spec/domains/diagram.yaml` from the verified IR, lays it out
and writes SVG and JSON. Every node sits in one root scope here; scopes, composites and pipeline
stages come with `story:diagram-zoom` and `story:diagram-pipelines`.

## What it adds

- `crates/modelspec-diagram/src/graph.rs`: `DiagramGraph` built from
  `modelspec_core::compile::compile_training` output (`crates/modelspec-core/src/compile.rs:54`):
  one Operator node per `TensorNode` of the model view's forward slice (the nodes
  `LoweredProgram.prediction` reaches, `crates/modelspec-core/src/lower.rs:31`), one edge per
  operand with the producer's dtype and shape; `graph_id`, `source_hash` and `ir_hash` as the
  domain says (`ir_hash` computed as `crates/modelspec-core/src/artifact.rs:156` computes it). Each
  node's glyph is `glyph::glyph_of` with the `InputRole` the builder classifies.
- `src/layout.rs`: the layered layout of DESIGN.md § 13.5 (chips, longest-path ranks, dummy nodes,
  four barycenter sweeps in exact integer arithmetic, integer coordinates).
- `src/svg.rs`: SVG that uses the glyph sprite's symbols (only those it needs, copied in, so a
  standalone file shows in a Markdown image), one
  `<g class="node" data-key data-op data-family data-dtype data-shape>` per visible node holding
  `<use href="#ms-…">`, the label and a `<title>` with the line `modelspec ir` prints for it; one
  `<path class="edge" data-from data-to data-operand>` per route; `role="img"`, `<title>`, `<desc>`.
- `src/json.rs`: `DiagramGraph` and `DiagramRender` as JSON (serde), field names as the domain.
- `src/model.rs`: the Rust types of the domain this story owns (`DiagramGraph`, `DiagramRender`,
  `DiagramNode`, `DiagramEdge`, `Scope`, `ScopeSegment`, `ScopeKey`, `ScopeKind`, `Port`,
  `PortDirection`, `NodeKind`, `DiagramView`, `NodeBox`, `EdgeRoute` and the ids), hand-written
  beside the generated schema and held to it by acceptance 4 (the repository's practice for
  `modelspec-observatory`). Each derives `Serialize`, `Deserialize` and
  `#[cfg_attr(feature = "ts", derive(ts_rs::TS))]`. `GlyphId`, `OperatorFamily`, `PipelineStage` and
  `EndpointRole` are re-exported from `glyph.rs` (`story:diagram-glyph-set`), never redefined.

## Scope

- `crates/modelspec-diagram/src/graph.rs`, `layout.rs`, `svg.rs`, `json.rs`, `model.rs`,
  `tests/render.rs` (new); `src/lib.rs` and `Cargo.toml` (add `serde_json`; cited from
  `story:diagram-glyph-set`), `Cargo.lock`.
- `crates/modelspec-diagram/schema/` (generated) and `Taskfile.yml` `schema-generate` and
  `schema-check` (one `--out ../crates/modelspec-diagram` line each).
- Not `crates/modelspec-core/**`, not `crates/modelspec-docs/**`, not `website/`, not
  `src/glyph.rs`.

## Acceptance

Tests in `crates/modelspec-diagram/tests/render.rs` (`cargo test -p modelspec-diagram`, `task ci`):

1. `diagram.model-matches-ir`: for `crates/modelspec-docs/models/two_layer_relu.mspec`, the model
   graph has one Operator node per forward-slice definition that `modelspec ir --stage tensor`
   prints, each with that line's op, dtype and shape, and one edge per operand; no `Mse` node and no
   node of the target `y`.
2. `diagram.deterministic`: rendering the same program twice, and a clone of it, gives equal SVG and
   JSON bytes; no numeric attribute of `x`, `y`, `width`, `height` or `d` holds a `.`.
3. `diagram.layered`: in every render of the three files of `examples/` that `task check-examples`
   checks and of `two_layer_relu.mspec`, every route goes from a lower `rank` to a higher one, no two
   boxes overlap, and no route passes through a box other than its ends.
4. `diagram.json-validates`: the graph and render JSON of those files validate against
   `crates/modelspec-diagram/schema/entities/modelspec.diagram.DiagramGraph.schema.json` and
   `…DiagramRender.schema.json`.
5. `diagram.chips`: in the model view of `two_layer_relu.mspec` the two `ParamRead`s are chips on
   their `MatMul`s, not ranked nodes, and the visible ranked nodes are `Input`, `MatMul`, `Relu`,
   `MatMul`.
6. `diagram.standalone-svg`: every `<use href="#ms-…">` of a rendered SVG names a `<symbol>` inside
   the same file.
7. `task schema-check` passes with the new `--out` line, and
   `cargo check -p modelspec-diagram --features ts` exits 0.

## Out of scope

- Composites, ports, zoom, depth (`story:diagram-zoom`); pipeline views (`story:diagram-pipelines`).
- Any page or SVG file on the site (`story:ir-gallery-atoms`).
- Kernel IR and execution plan diagrams.
