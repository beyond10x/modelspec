---
format: aep.planning-md/3
id: review-result:plan-critic-parallel-safety-ir-svg-gallery-round-1
kind: review-result
status: active
title: Plan critic parallel-safety, epic:ir-svg-gallery, round 1
summary: plan-critic-parallel-safety verdict on the architecture-diagrams decomposition, round 1
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
story:ir-gallery-modules — its `website/docs/diagrams/**` outputs are "the diagram pages of the examples not already written by `story:diagram-docs-blocks`", and `modules.mdx` links to those pages, but no `depends_on` edge runs from it to `story:diagram-docs-blocks`; add the edge or merge the two (cited, both bodies; also shares `website/static/img/diagrams` and `website/data/diagrams`) — .engineering/planning/story/ir-gallery-modules.md:50, .engineering/planning/story/ir-gallery-modules.md:61, .engineering/planning/story/diagram-docs-blocks.md:72
story:diagram-docs-blocks — acceptance 1 and 5 range over every fence under `website/docs` and every generated SVG, so they read files `story:ir-gallery-layers` creates (`layers.mdx` shows "its source", and the layer SVGs). Nothing orders the two and the manifest of 12 blocks omits them. Add an edge or scope both checks to the manifest's programs (inferred: the layers body says "its source", not that it is a fence) — .engineering/planning/story/diagram-docs-blocks.md:94, .engineering/planning/story/diagram-docs-blocks.md:105, .engineering/planning/story/ir-gallery-layers.md:60
story:observatory-architecture-graph — it generates `ui/src/generated/diagram.ts` from `modelspec-diagram` "(`ts` feature)", which needs `ts-rs` and a `ts` feature in `crates/modelspec-diagram/Cargo.toml` and derives in `src/model.rs`. Its scope lists neither file, and `story:ir-gallery-renderer` adds only `serde` and `serde_json`. The surface is unassigned and lands on the renderer's files. Either name an owner or add an ordering edge that records the shared file (cited; the files do not exist yet) — .engineering/planning/story/observatory-architecture-graph.md:63, DESIGN.md:777, .engineering/planning/story/ir-gallery-renderer.md:67
story:diagram-cli — adding `modelspec-diagram` to `crates/modelspec-cli/Cargo.toml` rewrites `Cargo.lock`, as `story:ir-gallery-atoms` does for `modelspec-docs`. The two have no ordering path and neither lists `Cargo.lock` (only `story:diagram-glyph-set` does). Either list the file in both and order them, or split the lock change out (inferred from the dependency additions; Cargo.lock is committed) — .engineering/planning/story/diagram-cli.md:52, .engineering/planning/story/ir-gallery-atoms.md:74

What I read: 12 artifacts (11 under `epic:ir-svg-gallery` plus `story:observatory-architecture-graph`), with `aep plan artifact show` on the epic and each story, `aep plan artifact waves --kind story --status draft`, and DESIGN.md § 13 (13.8 for the crate list). I also checked the tree with `git grep`, `ls` and `git status`.

What I could not establish:
- Surfaces. All 12 bodies name their paths, so all 12 are cited. 0 are inferred and 0 are unplaceable. The typed scopes tag `crates/modelspec-diagram/**` and the website outputs "inferred" only because those files do not exist yet.
- Acceptance reads. I traced 18 reads to a producer in the set, and 16 have a direct `depends_on` edge. The 2 without one are the first two findings. Reads of `story:decider-infer-on-data` are outside the set.
- Collisions with stories outside the set, not guessed and not counted toward the verdict:
  - `story:diagram-scope-provenance` collides with many decider stories on `lower.rs`, `autodiff.rs`, `ir/tensor.rs`, `ir/training.rs` and `ir/verify.rs`. Its body names only `story:decider-model-function-calls`.
  - `story:observatory-architecture-graph` collides on `server.rs` and `train.rs` with several observatory and vulkan bug stories.
- Out of lane:
  - `story:diagram-glyph-set` creates `OperatorFamily`, `PipelineStage` and `EndpointRole` use in `glyph.rs`, while `story:ir-gallery-renderer` hand-writes the domain types in `model.rs`, so type ownership is unstated (design lane).
  - Composite port rows in `story:diagram-zoom` need box sizes from `layout.rs`, which is not in its scope (design lane).
- Accepted as declared: the `CHANGELOG.md` appends in 7 stories are called "one line" or "append-only", and `gallery/mod.rs` is named as shared by `story:ir-gallery-layers` and `story:ir-gallery-modules`. Neither is a finding.
- The `depends_on` chain glyph-set → renderer → zoom → pipelines (with scope-provenance → zoom) orders every other overlapping pair in the set, including the `crates/modelspec-diagram` files and `crates/modelspec-docs` registration.

```findings
- file: .engineering/planning/story/ir-gallery-modules.md
  line: 61
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "its website/docs/diagrams/** outputs are the pages 'not already written by story:diagram-docs-blocks' and modules.mdx links to them, but no depends_on edge runs from this story to story:diagram-docs-blocks; add the edge or merge the two (cited, both bodies; shared website/static/img/diagrams and website/data/diagrams)"
- file: .engineering/planning/story/diagram-docs-blocks.md
  line: 94
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "acceptance 1 (every fence under website/docs) and 5 (every generated SVG) read files story:ir-gallery-layers creates (layers.mdx source, layer SVGs) with no ordering edge and absent from the 12-block manifest; add an edge or scope both checks to the manifest's programs (inferred: the layers body says 'its source', not that it is a fence)"
- file: .engineering/planning/story/observatory-architecture-graph.md
  line: 63
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "generating ui/src/generated/diagram.ts from modelspec-diagram's ts feature needs edits to crates/modelspec-diagram/Cargo.toml and src/model.rs, which are in no scope of this story and which story:ir-gallery-renderer owns (it adds only serde and serde_json); name an owner or an ordering edge recording the shared file (cited; the files do not exist yet)"
- file: .engineering/planning/story/diagram-cli.md
  line: 52
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "adding modelspec-diagram to crates/modelspec-cli/Cargo.toml rewrites Cargo.lock as story:ir-gallery-atoms does for modelspec-docs, the two have no ordering path and neither lists Cargo.lock; list it in both and order them, or split the lock change out (inferred from the dependency additions; Cargo.lock is committed)"
```
