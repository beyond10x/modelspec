---
format: aep.planning-md/3
id: story:vulkan-embedding-sort-lanes
kind: story
status: implemented
title: Embedding-gradient sort runs 16 lanes per id
summary: The EmbeddingGrad sort on Vulkan runs 16 invocations per id, each counting a stride of the staged block, the counts added in ascending lane order; 177 to 39 us in the BERT-tiny step.
relations:
- decomposes: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-11-04T22:28:18Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-11-04T22:28:19Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-11-05T15:49:21Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# Embedding-gradient sort runs 16 lanes per id

## Context

Profile 2026-11-04 after story:vulkan-long-k-from-4096, `train examples/bert_tiny.mspec` at 200
steps over packed-128, `--backend vulkan --profile`: steady 5,359 µs per step. The `per_row` sort
of `EmbeddingGrad` over the token table (dispatch position 261, 4,096 ids, 64 workgroups) took
177 µs at 0.28 GB/s: one invocation per id read every staged block of 64 ids serially, and 64
workgroups left most of the GPU idle.

## Outcome

The sort runs `schedule::EMBEDDING_SORT_LANES` (16) invocations per id, `n · 16` in all. Lane `l`
of id `n` counts, over each staged block of 64 ids, the ids at the block's positions `l`,
`l + 16`, `l + 32` and `l + 48` that sort before its own (smaller as `u32`, or equal at an
earlier position); after the block's barrier the 16 counts are staged and lane 0 adds them in
ascending lane order into the rank, and lane 0 writes the id and its position into row `rank`.
The rank is a count of integers, so its value is the same in any order; the sorted buffer is
bit-identical to the one-invocation sort's. The gradient kernel is unchanged.

## Acceptance

- `cargo test -p modelspec-runtime --features vulkan --test spirv_bert_rowwise --test
  adversary_rowwise --test spirv_val`: the sort kernel has `n · 16` invocations in workgroups of
  64 and passes `spirv-val`.
- `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test
  adversary_embgrad_sorted --test vulkan_bert_rowwise --test adversary_rowwise_device --
  --ignored`: the sorted ids and the gradient agree with the CPU interpreter, no validation
  message.
- The 200-step profile above, re-run: position 261 under 60 µs.

## Result

Profile 2026-11-04 on the changed tree, same command: position 261 takes 39 µs, steady 5,346 µs
per step. Loss at step 200: 8.047095 (CPU-tolerance class).
