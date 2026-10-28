---
format: aep.planning-md/3
id: review-result:plan-critic-design-ir-svg-gallery-round-1
kind: review-result
status: active
title: Plan critic design, epic:ir-svg-gallery, round 1
summary: plan-critic-design verdict on the architecture-diagrams decomposition, round 1
relations:
- reviews: epic:ir-svg-gallery
- reviews: story:diagram-glyph-set
- reviews: story:diagram-scope-provenance
- reviews: story:ir-gallery-renderer
- reviews: story:ir-gallery-atoms
- reviews: story:diagram-zoom
- reviews: story:diagram-pipelines
- reviews: story:ir-gallery-layers
- reviews: story:diagram-docs-blocks
- reviews: story:diagram-cli
- reviews: story:ir-gallery-modules
- reviews: story:observatory-architecture-graph
- reviews: story:diagram-infer-function-view
revision: 1
---
needs-revision
story:diagram-glyph-set — its API and acceptance 3 name `GlyphId`, `OperatorFamily`, `PipelineStage` and `EndpointRole` (and an `InputRole` that no spec, DESIGN.md section or crate defines), but its scope has no file that defines them, and story:ir-gallery-renderer, which depends on it, creates the domain types in `src/model.rs`; the seam is reversed, so the body must say that glyph-set owns these enums and `InputRole`, and the renderer's `model.rs` re-uses them — .engineering/planning/story/diagram-glyph-set.md:42, .engineering/planning/story/diagram-glyph-set.md:46, .engineering/planning/story/ir-gallery-renderer.md:61
story:ir-gallery-modules — its acceptance 1 and 4 (`task website` with `onBrokenLinks: throw`) read the per-program pages `website/docs/diagrams/&lt;program&gt;.mdx` that story:diagram-docs-blocks' `blocks.rs` writes, its scope names no code that writes them, and no `depends_on` edge runs from it to story:diagram-docs-blocks, so `aep plan artifact waves` can put them side by side; add `depends_on story:diagram-docs-blocks` — .engineering/planning/story/ir-gallery-modules.md:50, .engineering/planning/story/ir-gallery-modules.md:61, .engineering/planning/story/diagram-docs-blocks.md:72
story:observatory-architecture-graph — it generates `ui/src/generated/diagram.ts` from the `modelspec-diagram` types "(`ts` feature)", but no story in the set adds `ts-rs` or that feature to `crates/modelspec-diagram/Cargo.toml` or `model.rs`, and this story's scope lists neither file; the body must add them to its scope (or story:ir-gallery-renderer must add the feature with the types it owns) — .engineering/planning/story/observatory-architecture-graph.md:63, DESIGN.md:§ 13.8

What I read: 12 artifacts (epic:ir-svg-gallery and the 12 stories) in full via `aep plan artifact show`, plus `aep plan artifact relations`, `aep plan artifact graph`, `aep plan artifact validate`, DESIGN.md § 13.8, `spec/domains/diagram.yaml` and `crates/modelspec-observatory`. I walked all 152 `depends_on` edges in the store, including those leaving the set (for example to story:decider-infer-on-data, story:observatory-ui-live-loss-chart and story:observatory-parameter-statistics), and `tsort` found no cycle. I traced about 16 acceptance reads to a producer inside the set. All but the modules read of the diagram pages are recorded by an edge, directly or through the chain renderer, zoom and pipelines. The validator reports `valid`.

What I could not establish:
- Out of my lane, parallel safety: story:ir-gallery-modules and story:diagram-docs-blocks both write `website/docs/diagrams/**` and `website/data/diagrams/**`, and both edit `gallery/mod.rs` with story:ir-gallery-layers; the one-line, merge-in-either-order note covers only `mod.rs`.
- Out of my lane, acceptance: story:diagram-scope-provenance acceptance 2 is conditional on story:decider-model-function-calls, which is outside the set.
- I did not check whether `ts-rs` is already a workspace dependency usable by the new crate; the observatory crate takes it as `ts-rs.workspace = true`.

```findings
- file: .engineering/planning/story/diagram-glyph-set.md
  line: 42
  category: design
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "its API and acceptance 3 name GlyphId, OperatorFamily, PipelineStage, EndpointRole and an InputRole defined nowhere, while story:ir-gallery-renderer, which depends on it, creates the domain types in src/model.rs; the seam is reversed, so the body must say glyph-set owns these enums and InputRole and the renderer's model.rs re-uses them"
- file: .engineering/planning/story/ir-gallery-modules.md
  line: 50
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its acceptance 1 and 4 read the per-program diagram pages that story:diagram-docs-blocks' blocks.rs writes, its scope names no code that writes them, and no depends_on edge runs to story:diagram-docs-blocks; add depends_on story:diagram-docs-blocks"
- file: .engineering/planning/story/observatory-architecture-graph.md
  line: 63
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "it generates ui/src/generated/diagram.ts from modelspec-diagram types behind a ts feature that no story in the set adds, and its scope lists neither crates/modelspec-diagram/Cargo.toml nor src/model.rs; add them to its scope or have story:ir-gallery-renderer add the feature with the types it owns"
```
