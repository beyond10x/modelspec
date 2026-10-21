---
format: aep.planning-md/3
id: review-result:observatory-scope-round-2
kind: review-result
status: active
title: Plan critic scope, epic:training-observatory, round 2
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

story:observatory-live-event-stream — its Out of scope assigns "Filling `StepCompleted.metrics`" to `story:observatory-parameter-statistics`, and that story never mentions `metrics` (grep returns no match). The epic's "loss and metric curves" therefore has a dangling hand-off. The line should say that no M0 producer fills `metrics` and that `story:observatory-ui-live-loss-chart` exercises it with a fixture, as that story's Out of scope states — .engineering/planning/story/observatory-live-event-stream.md:111

story:observatory-architecture-graph — the epic promises the stream is replayable "so the UI also works after the run" (.engineering/planning/epic/training-observatory.md:32). This story's graph endpoint, scope and four scenarios cover only a live run. A recording holds `RunEvent`s only (`story:observatory-run-recording-replay`, Scope), so nothing says where the `ModelGraph` comes from in a replay with no training process. The body should either claim the replay case or record that the architecture view is live-only — .engineering/planning/story/observatory-architecture-graph.md:74

What I read: 9 artifacts (the epic and the 8 stories) via `aep plan artifact show`, plus `aep plan artifact graph`. I extracted 10 promises from the epic and traced all 10 to a story. Two traces carry the findings above (metrics and replay of the architecture view). The 10 are:
- the SSE stream with a serve flag
- replay from a recorded file
- the Rust HTTP server
- the Vue UI with generated types, drift-checked in `task ci`
- live loss and metric charts
- the architecture graph
- parameter and gradient statistics
- activation statistics
- one explainability method
- M0 examples on the CPU with no CPU-specific contract

All 4 epic Acceptance bullets are claimed: stream and replay by `story:observatory-live-event-stream`, the UI by `story:observatory-ui-live-loss-chart`, the architecture view by `story:observatory-architecture-graph`, and saliency by `story:observatory-saliency-gradient-x-input`.

Reach and overlap checks:
- `story:observatory-input-adjoint` is a compiler and DESIGN.md change. I judge it traceable to the epic sentence "gradient × input saliency first, since the compiler already derives gradients" (epic:39). The compiler derives gradients only for parameters, so the story is the prerequisite the sentence assumes. It does not reach beyond the parent.
- No two stories claim the same outcome. The adjoint's finite-difference test checks the compiler function and the saliency test checks the `Insight`.

What I could not establish:
- The epic says "9 `UNMAPPED:` markers" (epic:63), but `spec/domains/observatory.yaml` holds 10 and the stories settle all 10. This is a stale count in the epic, not a gap.
- Whether the DESIGN.md revision floor "0.7" in `story:observatory-input-adjoint` matches the current DESIGN.md is for the acceptance critic and out of my lane. Dependency ordering and shared-file safety are likewise out of my lane.

```findings
- file: .engineering/planning/story/observatory-live-event-stream.md
  line: 111
  category: scope
  severity: warning
  verdict: needs-revision
  origin: undecided
  message: "Out of scope assigns 'Filling StepCompleted.metrics' to story:observatory-parameter-statistics, which never mentions metrics, so the epic's 'loss and metric curves' has a dangling hand-off; the line should say no M0 producer fills metrics and the UI story covers it with a fixture"
- file: .engineering/planning/story/observatory-architecture-graph.md
  line: 74
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the epic promises the UI also works after the run (epic line 32) but this story's graph endpoint and scenarios cover only a live run and say nothing of where the ModelGraph comes from in a replay, since a recording holds RunEvents only; the body must claim the replay case or record that the view is live-only"
```
