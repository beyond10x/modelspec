---
format: aep.planning-md/3
id: review-result:plan-critic-acceptance-ir-svg-gallery-round-1
kind: review-result
status: active
title: Plan critic acceptance, epic:ir-svg-gallery, round 1
summary: plan-critic-acceptance verdict on the architecture-diagrams decomposition, round 1
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
story:diagram-scope-provenance — acceptance 2 is conditional on `story:decider-model-function-calls`, and before that story lands it is only "the one-level case of 1", so it passes on a tree with no nested-call support and nothing shows when nesting is covered; state the nested-call outcome as a fixture that exists now, or move it to the story that adds calls — .engineering/planning/story/diagram-scope-provenance.md:88
story:diagram-scope-provenance — acceptance 3 checks only that every `gradient` node has an `Adjoint(%k)` segment, but `story:diagram-pipelines` acceptance 2 and DESIGN.md § 13.4 need the forward-tagged rule values (`%9 Sub`, `%10 Scalar`, `%13`, `%15`, `%18 Transpose` in `modelspec ir --stage training` of `two_layer_relu.mspec`) to carry an Adjoint scope too, and no acceptance here says so — .engineering/planning/story/diagram-scope-provenance.md:91
story:observatory-architecture-graph — acceptance 7 "`task ui-types-check` exits 0, and `pnpm test` runs `ui/src/lib/diagram.test.ts`" reads the same before the work as after it, because `ui-types-check` exits 0 today and "runs" names no pass outcome; name that the check exits 1 while `ui/src/generated/diagram.ts` is missing or differs, and that `diagram.test.ts` passes — .engineering/planning/story/observatory-architecture-graph.md:116

What I read: all 12 ids (the 11 stories, plus `epic:ir-svg-gallery` and `story:observatory-architecture-graph` for context) with `aep plan artifact show`, plus `aep plan artifact kinds` and `aep plan artifact lifecycle story`. I checked citations in the tree: the Taskfile targets, `tensor.rs:800`, `lower.rs:234`, the `extra` test in `lower.rs`, the `ops::gradient` symbol, the 12 modelspec fences under `website/docs` (6 whole, 3 fragment, 3 skipped, which matches the table), the `diagram.yaml` port directions, and the `modelspec ir --stage training` print for `two_layer_relu.mspec`.

What I could not establish:
- Whether the `story:decider-infer-on-data` fixture that `story:diagram-infer-function-view` acceptance 1 names will exist, since that story is still draft.
- Which Update scope the shared learning-rate `Scalar` (`%20`) gets in `story:diagram-scope-provenance` acceptance 3. That is design, not acceptance, so it is out of my lane.
- Out of my lane, not counted in the verdict: `story:diagram-docs-blocks` acceptance 2 and `story:ir-gallery-atoms` acceptance 3 both claim the `two-layer-relu` page diagram, which is a shared-surface question. `story:diagram-pipelines` names the endpoint roles CheckpointSink and PredictionSink, while DESIGN.md § 13.4 and the domain say Checkpoint and Prediction. That is a design-lane question.

```findings
- file: .engineering/planning/story/diagram-scope-provenance.md
  line: 88
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "acceptance 2 is conditional on story:decider-model-function-calls and before it lands is only the one-level case of 1, so it passes without nested-call support and cannot be marked done on its own"
- file: .engineering/planning/story/diagram-scope-provenance.md
  line: 91
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "acceptance 3 checks only gradient-tagged nodes for an Adjoint segment, but story:diagram-pipelines acceptance 2 needs the forward-tagged rule values (%9, %10, %13, %15, %18) in an Adjoint scope and no acceptance here checks it"
- file: .engineering/planning/story/observatory-architecture-graph.md
  line: 116
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "acceptance 7 reads the same before and after the work: task ui-types-check exits 0 today and 'pnpm test runs diagram.test.ts' names no pass outcome"
```
