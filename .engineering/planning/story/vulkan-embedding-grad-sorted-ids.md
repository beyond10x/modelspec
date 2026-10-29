---
format: aep.planning-md/3
id: story:vulkan-embedding-grad-sorted-ids
kind: story
status: implemented
title: EmbeddingGrad over ids sorted once per batch
summary: The word-table gradient adds each occurring row from ids sorted once per batch, with the bits it has.
relations:
- decomposes: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
scope:
- confidence: inferred
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: inferred
  path: crates/modelspec-core/src/schedule.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T03:20:13Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T03:20:13Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-29T10:15:12Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"test_result":1}}}
---
# EmbeddingGrad over ids sorted once per batch

## Context

Profile 2026-10-29, perf/bert-step-tenth 9e448ada, `train examples/bert_tiny.mspec` 200 steps, packed-128, `--backend vulkan --profile`: steady 15.0 ms per step, 12.3 ms of it in kernels, 593 dispatches, 418 barriers and one submit and host wait per step. PyTorch eager 9.9 ms, torch.compile 4.9 ms (verification report of story:bert-tiny-vs-pytorch-cuda, training clock).

`EmbeddingGrad` takes 2.3 ms per step (24.6 % of kernel time). Since 0.70 each invocation owns one element of the table gradient and compares its row with all 4,096 ids of the batch (staged 64 at a time), so the word table ([8192, 128]) does 8192 × 128 × 4096 compares for 4,096 rows of real work.

## Outcome

The ids of a batch are sorted once on the device (stable, by id then by position), and each table row that occurs adds its gradient rows in ascending position order, so the bits are those of the current kernel; rows that do not occur write zero.

## Acceptance

- a probe first: the sorted form's kernel time on the profile above, against 2.3 ms;
- `vulkan_bert_tiny_same_bits` and `bert_tiny_vulkan_bits` unchanged; a new device test with repeated, absent and boundary ids (0, 8191) matches the CPU bit for bit, red first;
- interleaved A/B, 3 pairs, steady µs per step ratio recorded in the commit.
