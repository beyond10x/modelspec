---
format: aep.planning-md/3
id: review-result:bench-parallel-safety-round-2
kind: review-result
status: active
title: Plan critic parallel-safety, epic:transformer-vs-pytorch-cuda, round 2
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

Parallel-safety critic, round 2. All 3 round-1 findings are resolved; the second was only partly resolved, and its remaining part is new finding 2.

```findings
- file: .engineering/planning/story/benchmark-results-page.md
  line: 31
  category: parallel-safety
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the page reads verification-report:mnist-profile-throughput, which profile-achieved-throughput creates, and no edge orders the two"
- file: .engineering/planning/story/bert-tiny-vs-pytorch-cuda.md
  line: 25
  category: parallel-safety
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "it and transformer-block-benchmark both edit website/docs/benchmarks.md, and no edge orders them"
```
