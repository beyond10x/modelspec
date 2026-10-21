---
format: aep.planning-md/3
id: review-result:observatory-parallel-safety-round-1
kind: review-result
status: active
title: Plan critic parallel-safety, epic:training-observatory, round 1
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

- story:observatory-architecture-graph — it shares `crates/modelspec-observatory/`, `spec/domains/observatory.yaml` and `ui/src/generated/` (cited, all four bodies) with parameter-statistics, activation-statistics and saliency, and it names none of them. Its only edge is to the UI story, so arch/param, arch/activation and arch/saliency can run at once. Its acceptance 1 anchors on "line 221", which moves when any other settlement lands first. The body should record the shared surfaces by an ordering edge that names the file as its reason, or split the surfaces so they no longer overlap. — .engineering/planning/story/observatory-architecture-graph.md:52, .engineering/planning/story/observatory-architecture-graph.md:55, .engineering/planning/story/observatory-architecture-graph.md:57, .engineering/planning/story/observatory-parameter-statistics.md:56, .engineering/planning/story/observatory-activation-statistics.md:49, .engineering/planning/story/observatory-saliency-gradient-x-input.md:63
- story:observatory-saliency-gradient-x-input — it shares `crates/modelspec-observatory/`, `spec/domains/observatory.yaml` and `ui/src/generated/` (cited) with parameter-statistics and activation-statistics, and it names neither. It has no ordering edge to them, and its acceptance 1 anchors on lines 177 and 186, which shift when param (lines 29/56/79) or activation (line 81) settles first. The `ui/` shell file where each view is registered is inferred, because `ui/` does not exist yet. The body should record the shared surfaces by an ordering edge that names the file as its reason, or split the surfaces. — .engineering/planning/story/observatory-saliency-gradient-x-input.md:63, .engineering/planning/story/observatory-saliency-gradient-x-input.md:64, .engineering/planning/story/observatory-saliency-gradient-x-input.md:66, .engineering/planning/story/observatory-saliency-gradient-x-input.md:72, .engineering/planning/story/observatory-parameter-statistics.md:56
- story:observatory-saliency-gradient-x-input — it edits `crates/modelspec-core/src/autodiff.rs` and `compile.rs` to add an input adjoint, and does not say whether that lands in `TrainingProgram.program` or in a separate program. The Training IR outputs are "the loss followed by one gradient per entry of `gradients`, in `ParamId` order" (`crates/modelspec-core/src/ir/training.rs:5-6`). Parameter-statistics reads that layout and architecture-graph prints and tags it, and neither has an edge to saliency. The body should state that the training program's outputs and `ir --stage training` print are unchanged, or record the edge. — .engineering/planning/story/observatory-saliency-gradient-x-input.md:58, .engineering/planning/story/observatory-parameter-statistics.md:23-24, .engineering/planning/story/observatory-architecture-graph.md:72-73

What I read: 7 stories, the epic, `spec/domains/observatory.yaml` (markers at lines 5, 29, 56, 79, 81, 92, 177, 186, 221), `training.rs` and `autodiff.rs`. Commands: `aep plan artifact graph`, `aep plan artifact waves`, `cat` of each story and the epic, `grep` for symbols. No marker is claimed by two stories except line 5, which the two ordered halves split.

Establishment counts:

| Measure | Count |
|---|---|
| Items with surface cited | 7 |
| Items with surface inferred | 0 |
| Items not placed | 0 |
| Acceptance reads traced to a producer in the set | 9 |
| Of those, recorded by a `depends_on` edge | 9 |

Of the 9 recorded reads, 7 follow a direct edge and 2 follow a path through the UI story. Reads traced: replay→live, ui→live, ui→replay fixture, param→ui, activation→param (`Histogram`), activation→ui, arch→ui, saliency→ui, and param→live (stream).

What I could not establish:
- `aep plan artifact waves` lists all 7 as "unassessed" because the scope frontmatter is not written. This is expected, and the collisions above are what that frontmatter must declare.
- `ui/` and `crates/modelspec-observatory/` do not exist yet. Live-event-stream and ui-live-loss-chart create them, and every other story is ordered after both.
- The saliency story's `crates/modelspec-runtime/` is a whole crate. Whether it overlaps `runner.rs` and `optimizer.rs` (parameter-statistics, activation-statistics) is unknown. Those two are already ordered, so only saliency is exposed.
- Out of my lane, so not in the verdict:
  - The activation story's domain relation reads `GraphNode.value_id` from the architecture story's graph, with no edge (`observatory-activation-statistics.md:40-43`). It is conditional on the settlement, so it is not an acceptance read.
  - The UI story bundles the type generator, static serving, the headless browser harness and a CI Chromium it does not know about (`observatory-ui-live-loss-chart.md:54-56`). That is a design question.

```findings
- file: .engineering/planning/story/observatory-architecture-graph.md
  line: 52
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:observatory-architecture-graph shares crates/modelspec-observatory/, spec/domains/observatory.yaml and ui/src/generated/ (cited, all four bodies) with story:observatory-parameter-statistics, story:observatory-activation-statistics and story:observatory-saliency-gradient-x-input and names none of them; its only edge is to the UI story, so they can run at once, and its acceptance 1 anchors on line 221 of the shared yaml, which moves when another settlement lands first; record an ordering edge naming the shared file, or split the surface"
- file: .engineering/planning/story/observatory-saliency-gradient-x-input.md
  line: 63
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:observatory-saliency-gradient-x-input shares crates/modelspec-observatory/, spec/domains/observatory.yaml and ui/src/generated/ (cited) with story:observatory-parameter-statistics and story:observatory-activation-statistics and names neither, has no ordering edge to them, and anchors acceptance 1 on lines 177 and 186 which shift when they settle first; the ui/ shell file where views register is inferred because ui/ does not exist yet; record an ordering edge naming the shared file, or split the surface"
- file: .engineering/planning/story/observatory-saliency-gradient-x-input.md
  line: 58
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "story:observatory-saliency-gradient-x-input edits autodiff.rs and compile.rs to add an input adjoint without saying whether it lands in TrainingProgram.program, whose output layout (loss, then gradients in ParamId order, training.rs:5-6) story:observatory-parameter-statistics reads and story:observatory-architecture-graph prints and tags, and neither has an edge to it; state that the training program outputs and ir --stage training print are unchanged, or record the edge"
```
