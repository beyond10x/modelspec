---
format: aep.planning-md/3
id: story:transformer-block-benchmark
kind: story
status: draft
title: Transformer-block benchmark on synthetic data against PyTorch CUDA
relations:
- decomposes: epic:transformer-vs-pytorch-cuda
- depends_on: story:profile-achieved-throughput
- depends_on: story:bert-dense-gelu
- depends_on: story:bert-layer-norm-softmax
- depends_on: story:bert-attention
- depends_on: story:bert-vulkan-kernels-rowwise
- depends_on: story:bert-vulkan-kernels-attention-loss-adamw
- depends_on: story:mlp-size-sweep-vs-pytorch-cuda
- depends_on: story:benchmark-results-page
revision: 2
---
# Transformer-block benchmark on synthetic data

## Context

`epic:transformer-vs-pytorch-cuda`. The first transformer comparison: the two BERT-tiny encoder
blocks (hidden 128, 2 heads, sequence 128) on random `f32` inputs with an MSE loss and SGD. The
model itself needs no tokenizer, corpus, MLM batches, dropout or AdamW. The story still lands after
`story:bert-tiny-cpu-model`, because `story:bert-vulkan-kernels-rowwise` depends on it (shared
files, that story's body). The epic's measurement rules apply.

## Surface

- `examples/transformer_block.mspec`.
- A new Rust bench, `crates/modelspec-runtime/benches/transformer_block.rs`, with its `[[bench]]`
  entry in `crates/modelspec-runtime/Cargo.toml`, which `story:mlp-size-sweep-vs-pytorch-cuda` also
  edits (hence the `depends_on` edge to it).
- The block's table in `website/docs/benchmarks.md`, which `story:benchmark-results-page` creates.

## Acceptance

- `examples/transformer_block.mspec` passes `modelspec check` and trains on the CPU and on
  `--backend vulkan`; CPU and GPU losses agree within DESIGN.md § 9 tolerances over 10 steps.
- Every dispatch in the block's kernel plan has a FLOP and byte count; this story adds the
  counting rule for any kind that `train --profile` still prints as `uncounted`.
- `verification-report:transformer-block-benchmark-<date>` holds ModelSpec Vulkan, PyTorch eager
  and `torch.compile` µs per step, tokens per second and GFLOP/s (the same FLOP count for all three)
  at batch 8, 32 and 128 sequences, each the median of 3 runs.
- The report holds ModelSpec's and PyTorch's first-step losses on the same weights (exported from
  the ModelSpec initial checkpoint) and the same inputs, and they agree within the § 9 tolerance.
- The report names the host, the driver, the PyTorch version, both commands and the GPU clock
  during each run.
- `website/docs/benchmarks.md` shows the block's table, each figure linked to that report.
