---
format: aep.planning-md/3
id: story:bert-tiny-vs-pytorch-cuda
kind: story
status: active
title: BERT-tiny time to held-out loss against PyTorch CUDA
relations:
- decomposes: epic:transformer-vs-pytorch-cuda
- depends_on: story:bert-tiny-gpu-pretraining
- depends_on: story:profile-achieved-throughput
- depends_on: story:benchmark-results-page
- depends_on: story:transformer-block-benchmark
- serves: vision:modelspec
scope:
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-docs/src/benchmarks.rs
- confidence: cited
  path: crates/modelspec-runtime/benches
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/profile.rs
- confidence: cited
  path: opponents/bert_tiny_pretrain
- confidence: cited
  path: spec/domains/benchmark.yaml
revision: 11
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T07:33:52Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-28T07:33:53Z", actor: "human:timo", revision: 5}
---
# BERT-tiny time to held-out loss against PyTorch CUDA

## Context

`epic:transformer-vs-pytorch-cuda`. The real-life benchmark: the BERT-tiny pre-training run of
`story:bert-tiny-gpu-pretraining`, repeated in PyTorch on CUDA with the same corpus, tokenizer,
packed batches, initial weights and optimizer settings, and compared on wall time to the same
held-out MLM loss. The epic's measurement rules apply.

## Surface

- The run's table in `website/docs/benchmarks.md`, which `story:benchmark-results-page` creates.
- The kernel plan's FLOP and byte counting rules, for the kinds BERT-tiny dispatches that are still
  `uncounted` after `story:transformer-block-benchmark` (embedding, dropout, cross-entropy and AdamW
  where they are dispatched), and the DESIGN.md counting-rule table. That is the same surface as
  `story:profile-achieved-throughput` and the block story, hence the `depends_on` edge to the
  block story.
- The run uses `examples/bert_tiny.mspec` as `story:bert-tiny-gpu-pretraining` leaves it.

## Acceptance of the counts

- Every dispatch in BERT-tiny's kernel plan has a FLOP and byte count; `train --profile` prints no
  `uncounted` site for it, and a unit test checks each new kind's counts against hand-computed values.

## Acceptance

- Both sides evaluate held-out MLM loss every 100 steps on the same held-out split. Evaluation
  time is excluded from wall time, and the crossing step is read at that 100-step resolution.
- The target is the held-out unigram baseline that `story:bert-tiny-gpu-pretraining` records, and
  the report states its value.
- `verification-report:bert-tiny-vs-pytorch-<date>` holds, for ModelSpec Vulkan, PyTorch eager and
  `torch.compile`: µs per step, tokens per second, GFLOP/s (the same FLOP count for all three), and
  wall time until held-out MLM loss first falls below the target, each the median of 3 runs.
- The report gives both held-out loss curves and the step at which each side crosses the target.
- The report names the host, the driver, the PyTorch version, both commands and the GPU clock
  during each run.
- `website/docs/benchmarks.md` shows the run's table, each figure linked to that report.
