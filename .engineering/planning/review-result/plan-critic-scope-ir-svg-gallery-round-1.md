---
format: aep.planning-md/3
id: review-result:plan-critic-scope-ir-svg-gallery-round-1
kind: review-result
status: active
title: Plan critic scope, epic:ir-svg-gallery, round 1
summary: plan-critic-scope verdict on the architecture-diagrams decomposition, round 1
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

story:diagram-docs-blocks — the operator's request, "anywhere in the repo where in the docs we render code of model spec notation, we can also render the visual pendant of it", is narrowed to `website/docs`. `README.md:19` and `DESIGN.md:51` each hold a ` ```modelspec ` block, and no item draws them. The epic's Out of scope names neither file and the operator never excluded them, so the narrowing is unrecorded at the parent. The story's stated reason, "which GitHub renders without the site's component", does not hold: the renderer copies the glyphs it needs into each standalone SVG, so a committed file shows in a Markdown image. Either add the two blocks to this story's manifest, or record the exclusion in the epic's Out of scope as a decision. — `.engineering/planning/epic/ir-svg-gallery.md:42` (promise), `.engineering/planning/story/diagram-docs-blocks.md:114` (the exclusion)

Everything else the parent promises is claimed exactly once:
- Glyph per registry operator, and per operation the training step adds: `story:diagram-glyph-set`.
- Any model drawn from verified IR: `story:ir-gallery-renderer`.
- Scope table: `story:diagram-scope-provenance`.
- Zoom into composites: `story:diagram-zoom`.
- Train, eval and infer pipelines with stages that zoom into the model: `story:diagram-pipelines`.
- Command line: `story:diagram-cli`.
- Gallery atoms, layers, modules and pipelines: `story:ir-gallery-atoms`, `story:ir-gallery-layers`, `story:ir-gallery-modules`.
- Observatory UI drawing the same graph and zooming in place: `story:observatory-architecture-graph`.

Nothing in the set reaches beyond the parent. The three skipped docs blocks and the replay and live-overlay exclusions are named and match the epic's Out of scope. `story:diagram-infer-function-view` traces to the epic's "inference ... as pipelines" and sits in its landing order as "later".

What I read: 13 artifacts (the epic, the 11 stories, and `story:observatory-architecture-graph`) via `aep plan artifact show`. I also read `aep plan artifact graph`, the DESIGN.md § 13.6 and § 13.8 text, and `grep` for ` ```modelspec ` fences. I extracted 15 promises from the epic and traced 14 to an item.

What I could not establish:
- Out of my lane, did not set the verdict: `story:ir-gallery-modules` promises "each linking to its program's diagram page" for `mnist_mlp` and `transformer_block`. The story's acceptance does not check that those pages exist. That belongs to `plan-critic-acceptance`.
- Out of my lane, did not set the verdict: `story:ir-gallery-modules` and `story:diagram-docs-blocks` both write under `website/docs/diagrams/**`, and modules has no `depends_on` edge to docs-blocks. That belongs to `plan-critic-parallel-safety`.

```findings
- file: .engineering/planning/epic/ir-svg-gallery.md
  line: 42
  category: scope
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "story:diagram-docs-blocks — the operator's request, \"anywhere in the repo where in the docs we render code of model spec notation, we can also render the visual pendant of it\", is narrowed to website/docs; README.md:19 and DESIGN.md:51 each hold a modelspec block, no item draws them, and the epic's Out of scope does not exclude them; either add the two blocks to this story's manifest or record the exclusion in the epic's Out of scope as a decision (the story's own out-of-scope line is .engineering/planning/story/diagram-docs-blocks.md:114)"
```
