---
format: aep.planning-md/3
id: review-result:bench-design-round-1
kind: review-result
status: active
title: Plan critic design, epic:transformer-vs-pytorch-cuda, round 1
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

Design critic, round 1, over epic:transformer-vs-pytorch-cuda and its 5 draft stories.

- story:transformer-block-benchmark says it needs no tokenizer, corpus, MLM batches, dropout or AdamW. Its depends_on story:bert-vulkan-kernels-rowwise reaches story:bert-tiny-cpu-model, and that story pulls in the whole data pipeline. The edge exists because the two stories share files (bert-vulkan-kernels-rowwise.md:81).
- story:benchmark-results-page reads the block and BERT-tiny reports. It has an edge only to the sweep.
- story:profile-achieved-throughput counts matmul, elementwise and reduction dispatches. Nothing says who counts the dispatches the BERT kernel stories add.

```findings
- file: .engineering/planning/story/transformer-block-benchmark.md
  line: 23
  category: design
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the body claims an early benchmark with no data pipeline, but the depends_on chain through bert-vulkan-kernels-rowwise reaches bert-tiny-cpu-model and the whole data pipeline"
- file: .engineering/planning/story/benchmark-results-page.md
  line: 21
  category: design
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "reads the block and BERT-tiny reports with no edge to those stories, and neither producer carries the page update"
- file: .engineering/planning/story/profile-achieved-throughput.md
  line: 22
  category: design
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the counting rule covers matmul, elementwise and reduction only; no story owns the counts for layer_norm, softmax, attention, embedding or AdamW dispatches"
```
