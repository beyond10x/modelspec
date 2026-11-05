---
format: aep.planning-md/3
id: story:vulkan-matmul-split-k
kind: story
status: implemented
title: Split-K for long-k matmuls with a small output on Vulkan
summary: In a step of the tolerance class, a matmul with k from 4,096 below 16,384 and an output too small for a register-blocked tile is the partial products of slices of k and their sum in ascending slice order.
relations:
- decomposes: epic:bert-tiny-beats-torch-compile
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-11-05T10:32:08Z", actor: "human:timo", revision: 3, executor: "agent:modelspec-a4fb0b"}
- {from: "proposed", to: "active", at: "2026-11-05T10:32:08Z", actor: "human:timo", revision: 4, executor: "agent:modelspec-a4fb0b"}
- {from: "active", to: "implemented", at: "2026-11-05T15:49:22Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# Split-K for long-k matmuls with a small output on Vulkan

## Context

Profile 2026-11-05 at 5ca80f2e, `train examples/bert_tiny.mspec` at 200 steps over packed-128,
`--backend vulkan --profile` on the RTX 3090: steady 4,311.4 to 4,346.5 µs per step. The weight
gradients of the step are matmuls with a `k` of 4,096 (the 32 sequences of 128 tokens) or 8,192
(the vocabulary) and an output of at most 512 by 128, too small for a register-blocked tile, so
each took a long-`k` tile or one column per invocation and every output element was one chain
of `k` fused products in one invocation. A tile probe over these sites found no tile that keeps
that chain and is faster: at `[832, 8192] x [8192, 128]ᵀ` the 64 by 32 long-`k` block took 383
µs, the narrow one 425 and a 64 by 128 block 771.

## Outcome

In a step that holds an operator of the tolerance class of DESIGN.md § 5.4 item 2, the schedule
gives a matmul without an epilogue whose `k` is a multiple of 128 from 4,096 and below 16,384,
and whose output on the tiles of 128 invocations takes a long-`k` tile or one column per
invocation, two dispatches (`schedule::matmul_split_k_slices`, `MATMUL_SPLIT_K_FROM`,
`MATMUL_SPLIT_K_BELOW`, `MATMUL_SPLIT_K_SLICE`, `MATMUL_SPLIT_K_MAX_SLICES`):

- a `MatMul` kernel over the same operands whose result is `[P · m, n]`, `P = k / 128` slices
  halved while `P` is even and above 64: rows `s · m` and on hold the product over the slice's
  values of `k`, each element summed by one invocation in ascending `k` from the slice's first
  product, every later product fused (`Kernel::matmul_slices`, `spirv/emit.rs`);
- `SumRows` over the partial products as `[P, m · n]`, which adds the slices in ascending order.

Whether a matmul is split and into which slices is fixed by its shape and the step's operators;
a device profile changes only the tile of the first stage. The result is in the tolerance class.
A step without such an operator keeps every matmul whole with its bits, a matmul without slices
keeps its module word for word, and a `k` of 16,384 or more keeps the long-`k` tiles. DESIGN.md
0.84 states the kernel (§ 5.2), the class (§ 5.4) and the decision (ADR-010).

## Acceptance

- `cargo test -p modelspec-core`, `cargo test -p modelspec-runtime --features vulkan` and `cargo
  test -p modelspec-cli` pass; the split modules pass `spirv-val`.
- The plan of `examples/bert_tiny.mspec` has 13 split matmul dispatches, each read by one
  `SumRows` dispatch that writes the matmul's value; the plan scheduled within a device's limits
  splits the same matmuls into the same slices; a step without a tolerance-class operator and a
  matmul with an epilogue are not split.
- On the device, with `MODELSPEC_VK_VALIDATION=1`: the first stage has the bits of the CPU kernel
  over each slice, the second those of the ascending sum of the slices, and the sum is within the
  rounding bound of the CPU's unsplit matmul.
- Every device test of `modelspec-runtime` and `modelspec-cli` passes with no validation message,
  the recorded-bits tests of BERT-tiny after one re-recording.
- The 200-step profile above, re-run: at least 400 µs less per step.

## Result

Measured 2026-11-05 on the RTX 3090, `train examples/bert_tiny.mspec` at 200 steps over packed-128:

- Profiled, steady 3,674.2 to 3,759.5 µs per step (five runs, host load 11 to 27) against
  4,311.4 to 4,346.5 unsplit (two runs of the same binary with the split switched off, load 11 to
  13): 550 to 670 µs less. Slices of 256 took 3,719.0 to 3,739.0, of 512 3,789.8 to 3,802.1, of
  64 3,893.4 to 4,082.0, of 1,024 3,999.9 and of 2,048 4,381.8.
- 205 dispatches and 98 barriers per step where 5ca80f2e had 192 and 93; 13 split sites over
  packed data: `[832, 8192] x [8192, 128]ᵀ` in 64 slices, `[128, 4096]ᵀ x [4096, 128]` (8 sites),
  `[512, 4096]ᵀ x [4096, 128]` and `[128, 4096]ᵀ x [4096, 512]` (2 each) in 32.
- Without `--profile`, the losses of steps 0, 50, 100, 150 and 200 are 9.035598, 8.717873,
  8.329162, 8.119945 and 8.04707; 5ca80f2e printed 8.047078 to 8.047086 at step 200 in its
  profiled runs.
- `crates/modelspec-core/tests/matmul_split_k.rs` (6 tests) and
  `crates/modelspec-runtime/tests/vulkan_matmul_split_k.rs` (3 tests and the device test: 9
  shapes, 5,588,724 elements compared bit for bit with the CPU's AVX2 kernel) pass.
- Every `--ignored` target of `modelspec-runtime` (72) and `modelspec-cli` (35, the browser
  targets left out) ran in release with `MODELSPEC_VK_VALIDATION=1`. Outside tests ignored for a
  recorded finding or a missing data path, the failures were: the three recorded-bits tests of
  BERT-tiny (below); four matmul fixtures recorded on driver 2559459584, which refuse to compare
  on today's 2580660800 (`adversary_tiling_device`, `vulkan_matmul_same_bits`,
  `vulkan_matmul_plan_sites_same_bits`, `vulkan_matmul_tile256_same_bits`; their plans hold no
  tolerance-class operator, so no matmul of theirs is split); and two LayerNorm timing tests
  marked "the GPU to itself" that ran beside other tests, one ending in `ERROR_DEVICE_LOST`,
  which pass alone (`adversary_layer_norm_rows`, 4 of 4). `vulkan_heldout_metrics` passes with
  `MODELSPEC_BERT_TINY_PACKED` set (4 of 4).
- `vulkan_bert_tiny_same_bits`, `bert_tiny_vulkan_bits` and `bert_tiny_masked_rows` were recorded
  again: over packed data the loss of step 1 moves by one unit in the last place and 38 of the 41
  parameter digests change; on the dense data of the other two the four losses keep their bits
  and the parameter hash changes.
- `train --masked-rows`: the step of every row splits the weight gradient of the head's dense
  layer (`k` 4,096) and the step gathered at 768 rows splits the gradient through the vocabulary
  projection (`k` 8,192) instead, so the two no longer print the same parameter bits.
  `bert_tiny_masked_rows` holds the loss bits of both to one recording, records each run's
  parameters and compares all 1,486,720 parameters of one with the other within the element
  tolerance of DESIGN.md § 9 gate 5; the twin's gathered run still prints the ungathered run
  byte for byte. DESIGN.md § 8 and ADR-010 state it.

## Not done

- A `k` of 16,384 or more, the decider's weight gradients among them, is not split: split-K is
  not measured there. A later unit measures the decider's step with `MATMUL_SPLIT_K_BELOW`
  raised.
