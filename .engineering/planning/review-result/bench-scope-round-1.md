---
format: aep.planning-md/3
id: review-result:bench-scope-round-1
kind: review-result
status: active
title: Plan critic scope, epic:transformer-vs-pytorch-cuda, round 1
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

Scope critic, round 1, over epic:transformer-vs-pytorch-cuda and its 5 draft stories.

- The epic acceptance (epic line 66) asks every verification report to name the host, driver, PyTorch version, both commands, the median of at least 3 runs and the GPU clock during each run. No story claims anything beyond the median.
- story:transformer-block-benchmark reports achieved GFLOP/s for ModelSpec only. The epic promises "how much of the GPU each side used".
- story:bert-tiny-vs-pytorch-cuda has the same narrowing: no utilisation figure for PyTorch eager or torch.compile.

```findings
- file: .engineering/planning/epic/transformer-vs-pytorch-cuda.md
  line: 66
  category: scope
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "no story claims the report-content promise: host, driver, PyTorch version, both commands, GPU clock per run"
- file: .engineering/planning/story/transformer-block-benchmark.md
  line: 32
  category: scope
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "achieved GFLOP/s is reported for ModelSpec only; the epic promises GPU use for each side"
- file: .engineering/planning/story/bert-tiny-vs-pytorch-cuda.md
  line: 28
  category: scope
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "achieved GFLOP/s is reported for ModelSpec only; the PyTorch sides get no utilisation figure"
```
