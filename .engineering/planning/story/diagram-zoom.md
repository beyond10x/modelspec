---
format: aep.planning-md/3
id: story:diagram-zoom
kind: story
status: active
title: 'Zoom: composite nodes with ports, depth and expansion, one SVG per level linked by navigation'
summary: 'DiagramRender for a root, depth and expanded scopes: collapsed scopes as composite nodes with boundary ports, edges re-attached, one SVG per level whose composites link to their own level.'
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:ir-gallery-renderer
- depends_on: story:diagram-scope-provenance
scope:
- confidence: inferred
  path: crates/modelspec-diagram/src/graph.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/layout.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/svg.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/visible.rs
- confidence: inferred
  path: crates/modelspec-diagram/tests/fixtures/two_functions.mspec
- confidence: inferred
  path: crates/modelspec-diagram/tests/zoom.rs
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T22:02:03Z", actor: "human:timo", revision: 5}
- {from: "proposed", to: "active", at: "2026-10-29T22:02:03Z", actor: "human:timo", revision: 6}
---
# Zoom: composite nodes with ports, depth and expansion, one SVG per level linked by navigation

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.3. With the scope table of `story:diagram-scope-provenance`
every node has a scope path. This story turns scopes into zoom: a collapsed scope is one Composite
node with its own glyph and its boundary ports, and expanding it shows its children. It produces the
`modelspec.diagram.DiagramRender` of `spec/domains/diagram.yaml` for a root, a depth and a set of
expanded scopes, and the per-level SVG set whose composite nodes link to their own level.

## What it adds

- `crates/modelspec-diagram/src/graph.rs`: `Scope` entries from the IR's scope table (keys as the
  domain's `ScopeKey` says, `@c` from the second inlining as dropout sites are named), each with
  its ports, one per value crossing its boundary.
- `src/visible.rs`: the visible graph of a request `(root, depth, expanded)`: Operator nodes of
  expanded scopes, Composite nodes for collapsed ones, edges re-attached to ports, parallel edges
  merged with a `×n` label.
- `src/svg.rs`: a Composite node draws its `scope-*` glyph, its name and instance, its port row,
  and is an `<a>` whose `href` comes from a link template (`{scope}` replaced by the scope key);
  each level's root is drawn with a breadcrumb of its ancestors, each linked the same way.
- `levels(graph, template) -> Vec<(ScopeKey, String)>`: one SVG per scope at depth 0.

## Scope

- `crates/modelspec-diagram/src/graph.rs`, `src/svg.rs`, `src/layout.rs` (the composite box: its
  glyph, name and port row set its width, its height 56; cited from `story:ir-gallery-renderer`),
  `src/visible.rs`, `tests/zoom.rs`, `tests/fixtures/two_functions.mspec` (new).
- Not `crates/modelspec-core/**`, not `crates/modelspec-docs/**`.

## Acceptance

Tests in `crates/modelspec-diagram/tests/zoom.rs` (`cargo test -p modelspec-diagram`, `task ci`), on a
fixture `crates/modelspec-diagram/tests/fixtures/two_functions.mspec` whose loss calls `forward` and a
second function `head` twice (the shape the scope-provenance test uses):

1. `zoom.collapsed-is-one-node`: at depth 0 from the root, the render has exactly one Composite node
   per child scope (`forward`, `head`, `head@1`) and no Operator node inside them; each composite's
   ports equal the values crossing its boundary, and every visible edge ends on a visible node or
   a port.
2. `zoom.expand-shows-children`: with `head@1` in `expanded`, its Operator nodes are visible and
   `head` stays one node; the union of visible Operator nodes over all fully expanded renders equals
   the graph's Operator nodes.
3. `zoom.depth`: depth 1 from the root equals listing every child scope in `expanded`.
4. `zoom.links`: in `levels(graph, "{scope}.svg")` every composite's `href` names a level the call
   returned, and every breadcrumb link names an ancestor's level.
5. `zoom.deterministic`: the render JSON of one request is the same bytes twice and in any order of
   `expanded`.
6. `zoom.validates`: every render validates against the `DiagramRender` JSON Schema.

## Out of scope

- Pipeline stages, adjoint and update composites (`story:diagram-pipelines`).
- The observatory's in-place zoom (`story:observatory-architecture-graph`).
- Module and layer instances (DESIGN.md § 11 iteration 3).
