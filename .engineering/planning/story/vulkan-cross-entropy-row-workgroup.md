---
format: aep.planning-md/3
id: story:vulkan-cross-entropy-row-workgroup
kind: story
status: implemented
title: Cross-entropy row kernels run one workgroup per row
summary: The CrossEntropy row-loss and CrossEntropyGrad row-statistics kernels on Vulkan take one workgroup of 64 per row over the vocabulary, reading coalesced, in the tolerance class.
relations:
- decomposes: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-11-04T22:16:38Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-11-04T22:16:38Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-11-05T15:49:20Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# Cross-entropy row kernels run one workgroup per row

## Context

Profile 2026-11-04, origin/main acf24d85 (release 0.25.0), `train examples/bert_tiny.mspec` at
200 steps over packed-128, `--backend vulkan --profile`, warm-up and steps shortened to 20 and
200: steady 6,763 µs per step, 602 dispatches and 421 barriers per step. The two largest sites
were the `CrossEntropy` row-loss kernel and the `CrossEntropyGrad` row-statistics kernel over the
`[832, 8192]` logits of the gathered masked rows: 598 µs each, 45 GB/s, one invocation per row
reading 8,192 classes serially.

## Outcome

Each of the two kernels runs one workgroup of 64 per row. Invocation `i` takes classes `i`,
`i + 64`, ... in ascending order, so the workgroup's reads of a row are consecutive; the maxima
are staged and reduced in ascending invocation order by every invocation, the exponential sums
staged and added in ascending invocation order by invocation 0. The weight sum stays one
invocation, invocation 0 of the last workgroup. The row loss and the row statistics move from
the CPU's ascending class order to a two-stage order the shape alone fixes, which DESIGN.md
§ 5.4 places in the tolerance class (`exp` and `ln` already put them there); the maximum is the
same value in either order.

## Acceptance

- `cargo test -p modelspec-runtime --features vulkan --test spirv_bert_attention_loss_adamw
  --test spirv_cross_entropy_grad --test spirv_val`: the row kernels have one workgroup per row
  and pass `spirv-val`.
- `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test
  vulkan_bert_attention_loss_adamw --test adversary_gpu_step -- --ignored`: within tolerance of
  the CPU interpreter, no validation message.
- `cargo test --release -p modelspec-cli --test train_vulkan -- --ignored`: the BERT twin's
  Vulkan run agrees with the CPU run within the run tolerance.
- The 200-step profile above, re-run on the committed tree: both kernels under 100 µs.

## Result

Profile 2026-11-04 on the changed tree, same command: the two kernels take 62.9 and 62.7 µs
(434 GB/s), steady 5,715 µs per step against 6,763 before. Loss at step 200: 8.047056 against
8.047076 (CPU-tolerance class).
