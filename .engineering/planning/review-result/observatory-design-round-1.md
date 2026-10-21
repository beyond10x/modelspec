---
format: aep.planning-md/3
id: review-result:observatory-design-round-1
kind: review-result
status: active
title: Plan critic design, epic:training-observatory, round 1
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

- story:observatory-live-event-stream — its Scope lists four Rust types (`RunEvent`, `RunEventBody`, `StepCompleted`, `MetricSample`), but `StepCompleted.parameters` needs `ParameterStatistics`, `Histogram` and `StatisticsSubject` before it compiles or its RunEvent schema is generated, and `story:observatory-parameter-statistics` does not claim them either. Its Scope should name which story declares these three types, or say that live declares them with empty lists. — .engineering/planning/story/observatory-live-event-stream.md:62, .engineering/planning/story/observatory-live-event-stream.md:99, .engineering/planning/story/observatory-parameter-statistics.md:56, spec/domains/observatory.yaml:77
- story:observatory-saliency-gradient-x-input — it holds two separable things, a compiler design change and the Insight, endpoint and UI. Today the adjoint of an `Input` value is a change to `modelspec-core` `autodiff.rs` plus `DESIGN.md` § 3.3 and § 4. That change can be shown alone by a core gradient test and needs no UI, yet it waits behind `story:observatory-ui-live-loss-chart`. Split the adjoint into its own story that depends only on the M0 epic, and make saliency depend on it. — .engineering/planning/story/observatory-saliency-gradient-x-input.md:25, .engineering/planning/story/observatory-saliency-gradient-x-input.md:58

**What I read:** 8 artifacts (the epic and the 7 stories, whole bodies), `spec/domains/observatory.yaml` and `spec/domains/training.yaml:149`. I ran `aep plan artifact relations`, `graph` and `validate` (validate reports `valid`), and checked `runner.rs` and `optimizer.rs`.

**Graph:**
- **Walked edges:** I walked all declared edges, including those outside the set. There are 7 `depends_on` edges inside the set, 14 `decomposes` and `serves` edges to the epic and vision, and 3 outward edges from the epic.
- **Cycles:** I found none. `epic:bert-from-scratch` is `informed_by` this epic and nothing returns into the set.
- **Serialisation:** the set is not a queue. The edges form live, then replay, then UI, then a fan-out of architecture, parameters and saliency, then activations after parameters.
- **Acceptance reads:** I traced 11 acceptance reads of state a sibling produces, and an edge, direct or transitive, records every one:
  - the serve flag, the fixture recording, the headless harness and UI app;
  - `Histogram`, the step selector, the committed schema and the shared `runner.rs` observer.
- **All nine markers covered:** `UNMAPPED:` markers at lines 5, 29, 56, 79, 81, 92, 177, 186 and 221 are each settled by exactly one story, except line 5, which is deliberately split in halves between live and replay with an edge.

**What I could not establish:**
- **Activation keys (design):** if `story:observatory-activation-statistics` keys by `GraphNode.value_id`, its UI view may need the graph from `story:observatory-architecture-graph`. That depends on a settlement not yet made, so I raised no finding.
- **Marker line numbers (acceptance or parallel-safety lane):** every story cites marker lines such as 29, 56 and 177. Story 1 edits `observatory.yaml` first, so those numbers shift.
- **Final event (acceptance lane):** for the post-loop event (`k = steps`) `runner.rs` runs no `sgd_step`, so it has no gradient. `story:observatory-parameter-statistics` does not say what that event carries.
- **Shared files (parallel-safety lane):** the architecture, parameter, activation and saliency stories all edit `ui/` and `ui/src/generated/`.

```findings
- file: .engineering/planning/story/observatory-live-event-stream.md
  line: 62
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the Scope lists four Rust types but StepCompleted.parameters needs ParameterStatistics, Histogram and StatisticsSubject to compile and to generate the RunEvent schema, and story:observatory-parameter-statistics does not claim them either, so the body must say which story declares them"
- file: .engineering/planning/story/observatory-saliency-gradient-x-input.md
  line: 58
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the item holds a compiler design change (input adjoint in autodiff.rs and compile.rs, DESIGN.md 3.3 and 4) that a core gradient test shows alone and that needs no UI, yet it waits behind story:observatory-ui-live-loss-chart; split it into its own story depending only on epic:m0-semantic-core and make saliency depend on it"
```
