---
format: aep.planning-md/3
id: review-result:observatory-scope-round-1
kind: review-result
status: active
title: Plan critic scope, epic:training-observatory, round 1
relations:
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

- `story:observatory-ui-live-loss-chart` — the epic promises "The UI renders the linear-regression and two-layer-relu runs live and from a recording", but scenario 4 (`observatory.ui.replay`) replays only the linear-regression fixture, so the two-layer-relu recording case is narrowed with nothing saying it was dropped. The body should either add a two-layer-relu fixture recording and replay scenario, or state the narrowing as a decision. — .engineering/planning/story/observatory-ui-live-loss-chart.md:73 (promise quoted from .engineering/planning/epic/training-observatory.md:53; the fixture is linear-regression only at .engineering/planning/story/observatory-run-recording-replay.md:53)

**What you read:** 8 artifacts (the epic and its 7 stories) and the store graph, plus the markers in `spec/domains/observatory.yaml`.
- Commands: `aep plan artifact show` on each id, `aep plan artifact graph`, `grep -n UNMAPPED spec/domains/observatory.yaml`, and the epic file with line numbers.
- I extracted 15 promises from the epic (the 5 In-scope bullets split into 11 promises, plus 4 Acceptance bullets). 14 trace fully to a story and 1 traces partially (the finding above).
- All 9 `UNMAPPED:` markers (lines 5, 29, 56, 79, 81, 92, 177, 186, 221) are claimed. Line 5 is split live vs recorded-file across two stories, and the other eight are claimed once each.
- No story reaches into the epic's exclusions (no auth, no remote access, no comparison, no GPU).
- `epic:training-observatory` is the only store artifact that claims any of these promises. No older item covers them.

**What I could not establish:**
- The epic says "loss and metric curves", but no M0 producer fills `StepCompleted.metrics`, and the stories leave it empty. `story:observatory-ui-live-loss-chart` names this as out of scope and uses a fixture for the metric chart, so I did not count it as a gap.
- Out of my lane, so it does not set the verdict:
  - `story:observatory-saliency-gradient-x-input` shows that the epic's premise ("the compiler already derives gradients", epic line 39) is false for inputs. The story adds a compiler change and a DESIGN.md edit. The operator should see this, but it is traceable to the epic's saliency promise, so I did not call it reach.
  - Both `story:observatory-live-event-stream` and `story:observatory-run-recording-replay` replace the marker at line 5, which touches the same spec file and is parallel-safety's to judge.
  - The wording of the epic's `[Unreleased]` and CI additions (`ess specify validate` in `task ci`) is a design question.

```findings
- file: .engineering/planning/story/observatory-ui-live-loss-chart.md
  line: 73
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the epic promises \"The UI renders the linear-regression and two-layer-relu runs live and from a recording\" (epic line 53), but scenario 4 replays only the linear-regression fixture, so the two-layer-relu recording case is narrowed with nothing recording the decision"
```
