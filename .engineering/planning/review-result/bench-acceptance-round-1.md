---
format: aep.planning-md/3
id: review-result:bench-acceptance-round-1
kind: review-result
status: active
title: Plan critic acceptance, epic:transformer-vs-pytorch-cuda, round 1
relations:
- reviews: epic:transformer-vs-pytorch-cuda
- reviews: story:mlp-size-sweep-vs-pytorch-cuda
- reviews: story:profile-achieved-throughput
- reviews: story:transformer-block-benchmark
- reviews: story:bert-tiny-vs-pytorch-cuda
- reviews: story:benchmark-results-page
revision: 1
---
needs-revision

Acceptance critic, round 1, over epic:transformer-vs-pytorch-cuda and its 5 draft stories.

```findings
- file: .engineering/planning/story/benchmark-results-page.md
  line: 21
  category: acceptance
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "acceptance names `task check`, which Taskfile.yml does not define (website, docs-check and ci exist)"
- file: .engineering/planning/story/benchmark-results-page.md
  line: 23
  category: acceptance
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the pitch slide acceptance already holds today (slide 14, docs/pitch/index.html:526), so it names no transition"
- file: .engineering/planning/story/benchmark-results-page.md
  line: 25
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "same-change page update is a standing process rule, not an observable outcome"
- file: .engineering/planning/story/transformer-block-benchmark.md
  line: 33
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the report is not required to hold both first-step losses, and the block count is one or two"
- file: .engineering/planning/story/profile-achieved-throughput.md
  line: 26
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "'recorded in the story's evidence' names no artifact or location"
- file: .engineering/planning/story/profile-achieved-throughput.md
  line: 27
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the kernel-plan command is left open although --stage kernel exists (website/docs/reference/cli.md:107)"
- file: .engineering/planning/story/bert-tiny-vs-pytorch-cuda.md
  line: 26
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "time to target does not say how evaluation time is counted or fix the evaluation interval"
```
