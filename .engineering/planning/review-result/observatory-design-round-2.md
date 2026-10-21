---
format: aep.planning-md/3
id: review-result:observatory-design-round-2
kind: review-result
status: active
title: Plan critic design, epic:training-observatory, round 2
relations:
- reviews: story:observatory-input-adjoint
- reviews: story:observatory-live-event-stream
- reviews: story:observatory-run-recording-replay
- reviews: story:observatory-ui-live-loss-chart
- reviews: story:observatory-architecture-graph
- reviews: story:observatory-parameter-statistics
- reviews: story:observatory-activation-statistics
- reviews: story:observatory-saliency-gradient-x-input
revision: 1
---
needs-revision

Findings:

story:observatory-input-adjoint — the bullet "A way for a caller to build the target" names no function or signature for the `ReduceSum(prediction * mask)` target, and `story:observatory-saliency-gradient-x-input` is barred from `modelspec-core` while its `OutputElement { index }` target needs exactly that program. `TensorProgramBuilder` (`crates/modelspec-core/src/ir/tensor.rs:241`) has no constructor from a finished program, so nothing in the set owns this builder and the abstraction is split across both stories. The adjoint story should name the `modelspec-core` function that turns a forward program, its prediction and an output index into a rank-0 target, and its acceptance should call it. The saliency story should then say it calls that function. — .engineering/planning/story/observatory-input-adjoint.md:40, .engineering/planning/story/observatory-saliency-gradient-x-input.md:77

What I checked and found sound:
- **Cycles:** the graph has no cycle. The order is stream, then replay, then ui, parameter-statistics, architecture-graph, activation-statistics and saliency. The input adjoint hangs off `epic:m0-semantic-core` and joins at saliency.
- **Serial chain:** it is long, but every edge past ui has its reason written in the story (shared `observatory.yaml`, `crates/modelspec-observatory/`, `ui/src/generated/`, `runner.rs`; architecture-graph to activation-statistics also for `GraphNode.value_id`). Under the rubric that is a trade-off, not a finding. The alternative is splitting those shared surfaces.
- **Markers:** the 10 `UNMAPPED` markers in `spec/domains/observatory.yaml` each have exactly one owning story: 2 in stream, 1 in replay, 3 in parameter-statistics, 1 in architecture-graph, 1 in activation-statistics, 2 in saliency.
- **Stream and parameter-statistics seam:** the stream story declares the `Histogram` and `ParameterStatistics` types and sends them empty. Parameter-statistics fills them and settles their markers. This is a deliberate tracer-bullet seam and each half can be demonstrated alone.

What I read: 8 stories, `epic:training-observatory`, `spec/domains/observatory.yaml` markers and the code anchors the stories cite. Commands run: `aep plan artifact relations`, `graph` and `validate` (valid, 44 artifacts), and `show` for every id.

What I could not establish:
- **Edges walked:** 22 within the set, plus 4 outside it (`epic:m0-semantic-core`, the vision and the ESS document). No cycle.
- **Acceptance reads traced to a producer in the set:** 12. Eleven have a `depends_on` edge (headless-browser harness and fixtures to ui and replay, `value_id` to architecture-graph, the input-adjoint function to saliency). The twelfth is the finding above.
- **Out of my lane:** the epic says "9 `UNMAPPED:` markers" while the spec has 10 (scope). Saliency acceptance 2 evaluates "at the last step" while a recording holds no parameter values (acceptance, tied to marker 2). I did not read `spec/domains/training.yaml` beyond the relations the stories cite.

```findings
- file: .engineering/planning/story/observatory-input-adjoint.md
  line: 40
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the bullet 'A way for a caller to build the target' names no function or signature for the ReduceSum(prediction * mask) target, while story:observatory-saliency-gradient-x-input may not touch modelspec-core and needs that target for its OutputElement variant, so the construction is owned by neither story; the adjoint story should name the core function and test it, and the saliency story should say it calls it"
```
