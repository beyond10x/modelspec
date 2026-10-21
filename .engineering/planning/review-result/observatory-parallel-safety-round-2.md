---
format: aep.planning-md/3
id: review-result:observatory-parallel-safety-round-2
kind: review-result
status: active
title: Plan critic parallel-safety, epic:training-observatory, round 2
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

- story:observatory-input-adjoint — shares `CHANGELOG.md` with story:observatory-live-event-stream, story:observatory-run-recording-replay, story:observatory-ui-live-loss-chart, story:observatory-parameter-statistics, story:observatory-architecture-graph and story:observatory-activation-statistics (cited, all seven bodies). It has no `depends_on` path to any of them and its body doesn't mention the overlap. Either record the ordering edge with the shared file as its reason, or split the surface (for example, drop the `CHANGELOG.md` line from this story's Scope). — .engineering/planning/story/observatory-input-adjoint.md:53, .engineering/planning/story/observatory-live-event-stream.md:81

**What I read.** All 8 story bodies in full, plus `aep plan artifact graph`, `aep plan artifact waves`, `DESIGN.md`, `CHANGELOG.md`, and `crates/modelspec-docs/src/diagnostics.rs`.
- **Surfaces:** 8 cited from the `## Scope` sections, 0 inferred, 0 unplaceable. `waves` lists all 8 as unassessed because there is no `scope` frontmatter. That is known and not a finding.
- **Chain:** live-event-stream, then replay, then ui, then parameter-statistics, then architecture-graph, then activation-statistics, then saliency, with input-adjoint feeding saliency.
- **Acceptance reads traced to a producer:** 10 in the set. All 10 are recorded by a `depends_on` edge, 8 directly and 2 through the chain (parameter-statistics reading the `Histogram` types from live-event-stream, and saliency reading the UI).
- **Shared files inside the chain:**
  - `spec/domains/observatory.yaml`
  - `crates/modelspec-observatory/`
  - `ui/src/generated/`
  - `runner.rs` and `optimizer.rs`
  - `Taskfile.yml`, `AGENTS.md`, `CHANGELOG.md`, `website/docs/reference/`
  - Every pair is ordered by an edge, and the four-story shared-file order is stated in each body.
- **Input-adjoint:** it names its own independence ("needs nothing from the observatory stories", lines 24-25) and lists the files it does not touch (line 54). Its other surfaces are disjoint from the rest of the set.

**What I could not establish**
- **Out of lane, acceptance:** story:observatory-ui-live-loss-chart scenario 6 (line 83) needs a recording whose `StepCompleted` events carry an `accuracy` `MetricSample`. No Scope bullet says who writes that fixture, and M0 has no metric producer.
- **Inferred surface, unconfirmed:** story:observatory-input-adjoint refuses with a "typed error" (line 36). If that error is declared as an `E-` code, `modelspec-docs`' `diagnostics.rs` lists it, and `task docs-check` in `task ci` would need `website/docs/reference/diagnostics.md` regenerated. That file is not in its Scope. The body does not say whether the error is an `E-` code, so this is not a finding.

```findings
- file: .engineering/planning/story/observatory-input-adjoint.md
  line: 53
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:observatory-input-adjoint lists CHANGELOG.md in its Scope, a file also in the Scope of story:observatory-live-event-stream (observatory-live-event-stream.md:81) and the five later chain stories (cited, all bodies); it has no depends_on path to any of the six and does not mention the overlap, so record an ordering edge with the shared file as reason or split the surface"
```
