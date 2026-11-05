---
format: aep.planning-md/3
id: story:vulkan-attention-blocked-key-tiles
kind: story
status: active
title: Blocked key tiles for AttentionGradK and AttentionGradV on Vulkan
summary: AttentionGradK and AttentionGradV that share their call's softmax statistics run one workgroup per sequence, head and tile of 32 key rows, with the bits of the per-row kernel.
relations:
- decomposes: epic:bert-tiny-beats-torch-compile
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-11-05T08:59:57Z", actor: "human:timo", revision: 3, executor: "agent:modelspec-a4fb0b"}
- {from: "proposed", to: "active", at: "2026-11-05T08:59:57Z", actor: "human:timo", revision: 4, executor: "agent:modelspec-a4fb0b"}
---
# Blocked key tiles for AttentionGradK and AttentionGradV on Vulkan

## Context

Profile 2026-11-05 at 315a15ef, `train examples/bert_tiny.mspec` at 200 steps over packed-128,
`--backend vulkan --profile` on the RTX 3090: steady 5,246.7 µs per step. `AttentionGradK` and
`AttentionGradV` of each of the two layers ran one workgroup per key row: every workgroup of a
sequence recomputed the scores of its key against all 128 queries and loaded each `q`, `g` and `v`
row from device memory once per key row. The two layers' key-side gradients were 4 of the step's
dispatches; removing them from the command buffer took 130 to 280 µs off the step.

## Outcome

Where `AttentionGradK` and `AttentionGradV` share their call's softmax statistics under a key
mask, over sequences whose `S` and `L` are multiples of 32 and heads of `d` columns, `d` a
multiple of 16 of at most 64 (`schedule::attention_block_tile`, `ATTENTION_BLOCK_TILE`,
`ATTENTION_BLOCK_MAX_HEAD`), the schedule gives them one workgroup of 128 per sequence, head and
tile of 32 key rows, and the emitter `spirv/emit/attention_block.rs` writes the kernel: it loops
over the sequence's queries in chunks of 32, stages 16 columns at a time of the chunk's rows and
the tile's rows in workgroup memory, computes each score, probability and `dS` once per query and
key by the instructions of the per-row kernel, and adds each weighted row into its output sums in
ascending query order from `-0.0` with every product fused. Each element has the bits of the
per-row kernel; a masked key's row is `+0`. Any other shape keeps the per-row kernel. DESIGN.md
0.83 (§ 5.2) states the kernel.

## Acceptance

- `cargo test -p modelspec-core`, `cargo test -p modelspec-runtime --features vulkan` and `cargo
  test -p modelspec-cli` pass; the blocked modules pass `spirv-val`.
- The plan of `examples/bert_tiny.mspec` dispatches each `AttentionGradK` and `AttentionGradV`
  over 256 workgroups (32 sequences, 2 heads, 4 tiles).
- `train examples/bert_tiny.mspec --backend vulkan --data <packed-128>` at 200 steps prints the
  losses and parameters it printed before the change, bit for bit.
- `MODELSPEC_VK_VALIDATION=1` device tests `vulkan_bert_attention_loss_adamw`,
  `vulkan_attention_bits`, `vulkan_attention_prefix_bits` and `adversary_gpu_step` with
  `--ignored`: within tolerance of the CPU, no validation message.
- The 200-step profile above, re-run: at least 150 µs less per step.

## Result

Measured 2026-11-05 on the RTX 3090, `train examples/bert_tiny.mspec` at 200 steps over packed-128:

- Without `--profile`, the losses of steps 0, 50, 100, 150 and 200 and the digest of the `param`
  lines (`a4180f2c012d`) are those of 315a15ef.
- Profiled, steady 4,966.3 µs per step against 5,246.7 at 315a15ef, 280 µs less.
- Device tests with `MODELSPEC_VK_VALIDATION=1` and `--ignored`, release: `vulkan_attention_bits`
  (2), `vulkan_attention_prefix_bits` (1), `vulkan_attention_query_rows` (1),
  `vulkan_attention_spans` (2), `vulkan_bert_attention_loss_adamw` (2) and `adversary_gpu_step`
  (5) of `modelspec-runtime` pass with no validation message.
- `vulkan_attention_bits` gained
  `the_blocked_key_side_gradients_compute_the_bits_of_the_per_row_kernels`: on six further
  shapes (heads of 16, 32, 48 and 64 columns, 32 to 128 keys, a sequence masked whole) the
  blocked kernels run `b * heads * S / 32` workgroups and each of the four results has the hash
  of the kernel that runs alone, per row.
