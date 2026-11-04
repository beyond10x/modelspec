---
format: aep.planning-md/3
id: story:vulkan-long-k-from-4096
kind: story
status: active
title: Long-k matmul tiles from k of 4,096
summary: MATMUL_LONG_K_BLOCK and MATMUL_LONG_K_NARROW_BLOCK take outputs from k of 4,096 instead of 16,384; the BERT-tiny masked-row gradient 533 to 423 us, four weight gradients 128 to 86 us.
relations:
- decomposes: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-11-04T22:28:18Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-11-04T22:28:18Z", actor: "human:timo", revision: 3}
---
# Long-k matmul tiles from k of 4,096

## Context

Profile 2026-11-04 after story:vulkan-cross-entropy-row-workgroup, `train examples/bert_tiny.mspec`
at 200 steps over packed-128, `--backend vulkan --profile`: steady 5,715 µs per step. The masked-row
gradient `[832, 8192] x [8192, 128]ᵀ` (dispatch position 125) ran on the one-column tile at 533 µs
and 3.3 TFLOP/s, and the four weight gradients `[512, 4096]ᵀ x [4096, 128]` and
`[128, 4096]ᵀ x [4096, 512]` of the feed-forward layers on the two-row tile at 128 µs each; all five
have `k` of 4,096 or 8,192, below the long-`k` tiles' threshold of 16,384 (finding LK-1).

## Outcome

`MATMUL_LONG_K_BLOCK.from_k` and `MATMUL_LONG_K_NARROW_BLOCK.from_k` are 4,096. The tiles, their
bounds on rows and workgroups and the order of each element's sum are unchanged; only which outputs
they take moves. DESIGN.md § 5.2 and the tile docs in `ir/kernel.rs` state the threshold.

## Acceptance

- `cargo test -p modelspec-core` and `cargo test -p modelspec-runtime --features vulkan --test
  spirv_val`: the tile tests pass and every matmul module passes `spirv-val`.
- `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test
  adversary_gpu_step -- --ignored` and `cargo test --release -p modelspec-cli --test train_vulkan
  -- --ignored`: within tolerance, no validation message.
- The 200-step profile above, re-run: position 125 under 450 µs, the four weight gradients under
  100 µs each.

## Result

Profile 2026-11-04 on the changed tree, same command: position 125 takes 423 µs (4.1 TFLOP/s), the
four weight gradients 86 µs each (6.3 TFLOP/s), steady 5,359 µs per step against 5,715 before. An
experiment that also raised `MATMUL_WIDE_BLOCK.from_workgroups` to 256 moved no site and was
reverted.
