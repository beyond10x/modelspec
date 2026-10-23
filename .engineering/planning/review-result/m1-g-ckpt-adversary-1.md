---
format: aep.planning-md/3
id: review-result:m1-g-ckpt-adversary-1
kind: review-result
status: active
title: 'm1-g unit 4 adversary pass 1: 1 red (checkpoint write not atomic), 2 notes on weak GPU assertions'
relations:
- reviews: story:m1-vulkan-checkpoint-infer
revision: 1
---
```
unit: story:m1-vulkan-checkpoint-infer, tree modelspec-impl-m1g-ckpt at a40eca6 plus 2 untracked adversary test files
verdict: CONFIRMED (1 red, warning, in the shared checkpoint writer)
cases: executed 1324→1328, red 1
origin: introduced 0 / pre-existing 0 / undecided 1
```

1. **Cases added**:

| File | Case | On a40eca6 |
|---|---|---|
| cli `checkpoint_vulkan_adversary.rs` | `a_failed_checkpoint_write_keeps_the_checkpoint_it_replaces` | red: a failed write (EFBIG) left a 296-byte file where a 592-byte checkpoint was |
| cli | `infer_on_vulkan_refuses_every_bad_checkpoint_as_the_cpu_does` | green |
| cli | `train_on_vulkan_refuses_every_bad_resume_as_the_cpu_does` | green |
| cli | `checkpoint_every_usage_errors_on_vulkan` | green |
| cli, GPU | `vulkan_checkpoint_files_match_the_cpu_at_every_boundary` | green |
| cli, GPU | `vulkan_infer_prediction_matches_the_cpu` | green, bit-equal |
| cli, GPU | `profile_with_checkpoints_at_the_chunk_boundary` | green |
| runtime `vulkan_checkpoint_adversary.rs`, GPU | `checkpoint_boundaries_on_the_device` | green, bit-equal, 0 validation messages |
| runtime, GPU | `device_infer_prediction_matches_the_cpu_forward` | green, bit-equal, 0 validation messages |

2. **Suite**: `task ci` exit 201 at the red case; with `--no-fail-fast` 1327 passed, 1 failed, 110 ignored. GPU cases pass on the RTX 3090 with no other compute process.

3. **Attacked, not broken**: `--checkpoint-every` not dividing the steps, equal, larger, 1, 0 (exit 2), on the 256-step chunk boundary; resume from a non-multiple step; resume on Vulkan from a CPU checkpoint and from another file's (E-CKPT-003); infer with a missing, empty, truncated, payload-flipped or other-file checkpoint gives the CPU's E-CKPT code before the loader; `--profile` with checkpoints; mid-run checkpoint values bit-equal to a run of exactly k steps; DESIGN.md § 6 and § 8, cli.md and diagnostics.md against behaviour.

4. **Fix by the coordinator** in 6dea124: `checkpoint::write` writes `<path>.tmp`, syncs, renames.

```findings
- file: crates/modelspec-runtime/src/train/checkpoint.rs
  line: 605
  category: concurrency
  severity: warning
  verdict: CONFIRMED
  origin: undecided
  message: checkpoint::write truncates and rewrites in place, so a write that fails part-way (EFBIG/ENOSPC or a kill) destroys the checkpoint at --checkpoint, including the --resume source when both name one file, on both backends
- file: crates/modelspec-runtime/tests/vulkan_checkpoint.rs
  line: 206
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: mid-run device checkpoints are checked for step and shape only, so a runner writing stale or initial values at a checkpoint stays green
- file: crates/modelspec-runtime/tests/vulkan_checkpoint.rs
  line: 368
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: device infer's prediction is checked only for its rank, and the CLI test checks only the loss, so a wrong prediction read-back stays green
```
