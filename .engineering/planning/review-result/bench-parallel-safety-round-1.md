---
format: aep.planning-md/3
id: review-result:bench-parallel-safety-round-1
kind: review-result
status: active
title: Plan critic parallel-safety, epic:transformer-vs-pytorch-cuda, round 1
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

Parallel-safety critic, round 1, over epic:transformer-vs-pytorch-cuda and its 5 draft stories. All 5 have no scope, so `aep plan artifact waves` lists every one as unassessed.

```findings
- file: .engineering/planning/story/benchmark-results-page.md
  line: 21
  category: parallel-safety
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "reads the block and BERT-tiny reports, but its only depends_on edge runs to the sweep, so one wave could hold the page and two of its producers"
- file: .engineering/planning/story/benchmark-results-page.md
  line: 25
  category: parallel-safety
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the same-change rule makes three producers edit website/docs/benchmarks.md and docs/pitch/index.html unordered and unnamed"
- file: .engineering/planning/story/transformer-block-benchmark.md
  line: 30
  category: parallel-safety
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "names no measurement surface; if it and the sweep both add a bench beside crates/modelspec-runtime/benches/mnist_shape.rs, they collide on crates/modelspec-runtime/Cargo.toml (inferred)"
```
