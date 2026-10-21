---
format: aep.planning-md/3
id: review-result:observatory-acceptance-round-1
kind: review-result
status: active
title: Plan critic acceptance, epic:training-observatory, round 1
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

- story:observatory-live-event-stream — scenario 1 names the markers by line ("lines 5 (live half) and 92"), and line 5 is one comment block shared with the recording story, so "the live half is replaced" cannot be read off the file. The line numbers also stop being true once this story edits the file. Name the markers by their `UNMAPPED:` text or the entity they sit on — .engineering/planning/story/observatory-live-event-stream.md:80
- story:observatory-run-recording-replay — scenario 1 cites "the recorded-file half of the marker at line 5", but this story depends on the live story, whose settlement rewrites that line first. No marker is left at line 5 to check, and the half-comment has no stable name — .engineering/planning/story/observatory-run-recording-replay.md:59
- story:observatory-architecture-graph — scenario 1 cites "the marker at line 221"; `depends_on` (via the UI story and its predecessors) puts earlier settlements ahead of this one, so line 221 will have moved. Name it by the `UNMAPPED:` comment on `ModelGraph.artifact_id` — .engineering/planning/story/observatory-architecture-graph.md:64
- story:observatory-parameter-statistics — scenario 1 cites "lines 29, 56 and 79", which the earlier-landing live-stream and recording settlements shift. Name them by `Histogram`, `ParameterStatistics.norm` and `StepCompleted.parameters` — .engineering/planning/story/observatory-parameter-statistics.md:65
- story:observatory-activation-statistics — scenario 1 cites "the marker at line 81", which the chain of earlier settlements shifts. Name it by the `UNMAPPED:` comment on `StepCompleted` that lists activation statistics — .engineering/planning/story/observatory-activation-statistics.md:57
- story:observatory-saliency-gradient-x-input — scenario 1 cites "lines 177 and 186", which the chain of earlier settlements shifts. Name them by the `Insight` comment and the `Insight.training_run_id` comment — .engineering/planning/story/observatory-saliency-gradient-x-input.md:72
- story:observatory-parameter-statistics — scenario 2 compares "the last such event" to the final `param @LinearReg.w` line. That holds only if marker 3 settles on a sampling rule whose last sampled step is step 500, and the acceptance does not say so, so a valid settlement can fail it. Compare against the values at the step of that event, or require the final step to carry statistics — .engineering/planning/story/observatory-parameter-statistics.md:67
- story:observatory-ui-live-loss-chart — scenario 6 ("`task build` and `task test` pass with no `node` or `pnpm` on `PATH`") is not a named test and names no command that runs it. It is under a heading saying every scenario is run by `task ci`, but `task ci` now includes the pnpm `ui` task (line 51). It is also true before the work as well as after. Give the exact command, for example `task build test` under a `PATH` without node, and say that `task ci` does not run it — .engineering/planning/story/observatory-ui-live-loss-chart.md:78
- story:observatory-architecture-graph — scenario 4 says the Training stage "distinguishes Forward, Loss, Gradient and Optimizer nodes" but names nothing the headless browser reads (an attribute, class or label per subgraph). "Distinguishes" is a judgement, not an observation. Name the DOM attribute the test asserts per node — .engineering/planning/story/observatory-architecture-graph.md:75
- story:observatory-saliency-gradient-x-input — scenario 2 checks against "the attributed scalar", but the story says that choice (the loss or one model-output element) is not in the marker (line 40-41). Scenario 1 does not require it to be recorded anywhere. `Insight` has no field for it, so a second person cannot reproduce the finite-difference check. Require the settlement to state the scalar in a typed field or field summary, and scenario 2 to cite it — .engineering/planning/story/observatory-saliency-gradient-x-input.md:76

**What I read:**
- All 7 ids were read in full, with `aep plan artifact kinds`, `aep plan artifact lifecycle story`, and `cat -n` of each `.engineering/planning/story/observatory-*.md`.
- I also read the epic, `spec/domains/observatory.yaml` and the cited paths and tests in `crates/`.
- I confirmed the nine marker lines (5, 29, 56, 79, 81, 92, 177, 186, 221) and that `ess generate --kind schema` emits `RunEvent`, `Insight` and `ModelGraph` schemas.
- I ran `ess specify validate` (valid, ESS-ENTITY-019 warnings on `Insight` and `ModelGraph` are present today).

**What I could not establish:**
- Whether the lists of 4 to 6 scenarios per story should count as "more than one statement". Each story's done state is the plain conjunction of named tests, so I did not flag it. Say so if the rubric means one statement per story.
- Whether `ess` accepts `--check` with `--out` for the committed schema (live-stream scenario 5). I ran generation but not the check.
- Out of my lane, for the design and scope critics:
  - The saliency story's scope edits DESIGN.md § 3.3 and § 4 and the revision line, which no scenario observes.
  - The architecture story's scope derives the graph from `TrainingProgram`, while its scenario 4 needs Optimizer nodes, which live in `TrainingStep`.
  - The UI story's replay scenario covers the linear-regression fixture only, while the epic wants both runs from a recording.

```findings
- file: .engineering/planning/story/observatory-live-event-stream.md
  line: 80
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 1 names the markers by line number (lines 5 (live half) and 92); line 5 is one comment block shared with the recording story and the numbers move when this story edits the file, so name the markers by their UNMAPPED text or entity"
- file: .engineering/planning/story/observatory-run-recording-replay.md
  line: 59
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 1 cites the recorded-file half of the marker at line 5, which the depended-on live story's settlement has already rewritten, so no marker exists at that line to check; name it by its UNMAPPED text"
- file: .engineering/planning/story/observatory-architecture-graph.md
  line: 64
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 1 cites the marker at line 221, which earlier-landing settlements in the same file will have moved; name it by the UNMAPPED comment on ModelGraph.artifact_id"
- file: .engineering/planning/story/observatory-parameter-statistics.md
  line: 65
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 1 cites markers at lines 29, 56 and 79, which earlier-landing settlements shift; name them by Histogram, ParameterStatistics.norm and StepCompleted.parameters"
- file: .engineering/planning/story/observatory-activation-statistics.md
  line: 57
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 1 cites the marker at line 81, which earlier-landing settlements shift; name it by the UNMAPPED comment on StepCompleted about activation statistics"
- file: .engineering/planning/story/observatory-saliency-gradient-x-input.md
  line: 72
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 1 cites markers at lines 177 and 186, which earlier-landing settlements shift; name them by the Insight comment and the Insight.training_run_id comment"
- file: .engineering/planning/story/observatory-parameter-statistics.md
  line: 67
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 2 equates the last event carrying statistics with the final param line, which holds only if marker 3 settles on sampling that includes step 500, and the acceptance does not say so, so a valid settlement can fail it"
- file: .engineering/planning/story/observatory-ui-live-loss-chart.md
  line: 78
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 6 (task build and task test pass with no node or pnpm on PATH) is not a named test, names no command that runs it, sits under a heading saying task ci runs every scenario although task ci now includes the pnpm ui task, and reads the same before and after the work"
- file: .engineering/planning/story/observatory-architecture-graph.md
  line: 75
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 4 says the Training stage distinguishes Forward, Loss, Gradient and Optimizer nodes but names nothing the headless browser reads (attribute, class or label per subgraph), so the check is a judgement"
- file: .engineering/planning/story/observatory-saliency-gradient-x-input.md
  line: 76
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "scenario 2 checks against the attributed scalar, but the story says that choice is not in the marker (lines 40-41) and no scenario requires it recorded anywhere (Insight has no field for it), so the finite-difference check cannot be reproduced by anyone else"
```
