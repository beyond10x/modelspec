---
format: aep.planning-md/3
id: review-result:adversary-m1-vulkan-train-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-vulkan-train
relations:
- reviews: story:m1-vulkan-train
revision: 1
---
# Adversary pass 1, story:m1-vulkan-train

Unit `2a6637f` (impl/m1-vulkan-train), attacked in a scratch copy. Verdict: NEEDS-CHANGE.

## Mutants

| mutant | unit `vulkan_train --ignored` | adversary cases |
|---|---|---|
| M1 last step records the loss only, no update (`runner.rs:266`) | 2 passed | red |
| M2 no final forward, slot `steps` stale (`runner.rs:272`) | 2 passed | red |
| M3 learning rate x1.01 in the emitter | red | red |
| M4 no chunking, one command buffer (`runner.rs:261`) | 2 passed | red |
| planted `burn-cuda` dependency | `no_cuda` ok | red |

All GPU runs with validation on, RTX 3090 alone.

## New cases

- `crates/modelspec-runtime/tests/adversary_vulkan_train_chunks.rs`: linear regression at 0, 1, 255,
  256, 257, 512 steps against the CPU, submits equal to `max(1, ceil(steps/256))`, exact bytes;
  mnist at 257 steps closer to the CPU after 257 updates than after 256.
- `crates/modelspec-runtime/tests/adversary_no_cuda_contains.rs`: no crate name contains `cuda`,
  `cudarc`, `cust` or `nvrtc`.

## Attacked, not broken

Single host wait; 0, 1 and 256 boundaries; loss slot indexing; parameter versions; byte counters
(counted from recorded copies); a wrong learning rate; E-TARGET-002 creates no instance; no
`task test` process opens `libvulkan.so`, an ICD file or `/dev/nvidia*` (strace, with a positive
control).

```findings
- file: crates/modelspec-runtime/tests/vulkan_train.rs
  line: 118
  category: mutant
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: skipping the last step's update or the final forward pass leaves both unit GPU tests green, because at 500/1800 steps one update is inside the whole-run and loss bounds; adversary_vulkan_train_chunks.rs catches both
- file: crates/modelspec-runtime/tests/vulkan_train.rs
  line: 163
  category: mutant
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: submits is bounded only from above, so a runner that never chunks (1 submit for 1800 steps) passes although the story requires command buffers of at most 256 steps
- file: crates/modelspec-runtime/tests/no_cuda.rs
  line: 25
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: is_cuda_crate matches whole names or stems only, while acceptance 5 says names containing the words; a planted burn-cuda dependency passes no_cuda.rs and fails adversary_no_cuda_contains.rs
- file: crates/modelspec-runtime/tests/no_cuda.rs
  line: 133
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the readelf check inspects whatever modelspec binary an earlier build left in the profile directory, so it can pass on a stale binary or fail when the acceptance-5 command runs alone, and it cannot see a dlopen
- file: crates/modelspec-runtime/src/vulkan/runner.rs
  line: 243
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: RunStats::elapsed starts before the uploads and the first chunk are recorded, while its doc at line 62 says it starts at the first submit
```
