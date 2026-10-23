---
format: aep.planning-md/3
id: review-result:bench-design-round-2
kind: review-result
status: active
title: Plan critic design, epic:transformer-vs-pytorch-cuda, round 2
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

Design critic, round 2. Round-1 findings 1 and 2 are resolved. Finding 3 is partly resolved; its remainder is new finding 2.

```findings
- file: .engineering/planning/story/benchmark-results-page.md
  line: 31
  category: design
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "reads verification-report:mnist-profile-throughput with no depends_on edge to profile-achieved-throughput"
- file: .engineering/planning/story/bert-tiny-vs-pytorch-cuda.md
  line: 36
  category: design
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "no story owns counting rules for BERT-tiny's embedding, dropout, cross-entropy and AdamW dispatches, which the full FLOP count needs"
```
