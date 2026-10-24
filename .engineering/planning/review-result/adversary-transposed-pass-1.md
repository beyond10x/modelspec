---
format: aep.planning-md/3
id: review-result:adversary-transposed-pass-1
kind: review-result
status: active
title: Adversary pass 1 on matmul-reads-transposed-operand
relations:
- reviews: story:matmul-reads-transposed-operand
revision: 1
---

unit: story:matmul-reads-transposed-operand (worktree `<trees>/modelspec-impl-transposed`, HEAD f046123, base ecabada)
verdict: red: 1 case, ADV-T1, INFEASIBLE; no reachable defect
cases: executed 1306→1316, red 1 that is mine plus 13 that came before this unit
origin: introduced (all 3 findings about this unit); the 13 other red cases come from earlier waves or the environment
wrote-outside-worktree: `<wave-root>/transposed/adv1-scratch/` (`base/` and `mutant/` tree copies with their targets cleaned, `art-base-*`, `art-head-*`, `nofold.mspec`, `suite-workspace-release.log`)
needs-coordinator: no

**Diff stat.** `git --no-pager diff --stat` shows only `.engineering/planning/story/matmul-reads-transposed-operand.md | 7 +++++--`. That change is yours; I did not touch it, nor the untracked `matmul-tile-from-device-invocation-limit.md`. My files are all untracked:
- `<trees>/modelspec-impl-transposed/crates/modelspec-core/tests/adversary_transposed_fold.rs` (5 tests)
- `<trees>/modelspec-impl-transposed/crates/modelspec-core/tests/adversary_transposed_artifact.rs` (1 test passes, 1 is ADV-T1 and `#[ignore]`d)
- `<trees>/modelspec-impl-transposed/crates/modelspec-core/tests/fixtures/adversary_transposed_base_linear_regression/` (an artifact the base commit wrote)
- `<trees>/modelspec-impl-transposed/crates/modelspec-runtime/tests/adversary_transposed_spirv.rs`
- `<trees>/modelspec-impl-transposed/crates/modelspec-runtime/tests/adversary_transposed_device.rs` (2 device tests)

`cargo fmt` and clippy `-D warnings` are clean on both crates. No `/home/` path is in these files.

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Fold exclusions | `adversary_transposed_fold.rs` checks every Transpose in 8 programs: kept or folded, as counted | Passes. With the prediction exclusion at schedule.rs:266 deleted, 92 core cases stay green and `verify_plan` accepts the plan. Only my test catches it: "prediction_transpose: the loss or prediction ValueId(5) has no dispatch". Finding F2. The update exclusion cannot be reached from a source. |
| 2 | Operand positions, both operands transposed, every tile | fold test plus the device tests | Passes. Both operands flagged at tiles (1,1), (8,8) and (8,4). |
| 3 | Partial tiles | 15 shapes × 4 flag combinations in exactly sized buffers, against the CPU oracle | "validation true; 15 shapes x 4, 22765712 elements against the CPU oracle", same bits |
| 4 | Accumulation order, NoContraction | `adversary_transposed_spirv.rs`: 16 shapes, all 4 tile families, with and without epilogue | Passes. Arithmetic sequence the same as the plain module, every result NoContraction, spirv-val passes. |
| 5 | Workgroup memory, bank conflicts, coalescing | spirv test (array lengths equal) plus arithmetic, no test | Memory size unchanged, global reads coalesced. Bank conflicts: finding F3. |
| 6 | Artifact compatibility | base-commit fixture reads at head, hash 23cb…; ADV-T1 | A source with nothing to fold compiles byte-identical at base and head (`diff -r`). ADV-T1 is red, quoted below. |
| 7 | Signature and module-id separation | two kernels of the same shape, one flagged | Different ids and different printed lines (`Tensor<f32, [10, 3]> transposed`) |
| 8 | Guide, goldens, docs-check | `cargo run -p modelspec-docs -- generate --check` | EXIT=0. Guide numbers agree with the golden. |
| 9 | Dispatch, barriers, profile counts | golden plus the fold audit | `barriers_per_step` stays 10 because the folded transposes shared barrier phases. Not a defect. |
| 10 | Device failures with `task ui` built | `MODELSPEC_VK_VALIDATION=1 b10x-build-slot cargo test --release --workspace --no-fail-fast -- --include-ignored` (no process on the GPU before) | EXIT=101: 1302 passed, 14 failed. Breakdown below. |

**ADV-T1 output:** `panicked at crates/modelspec-core/tests/adversary_transposed_artifact.rs:106:9: a Elementwise kernel flagged transposed [true, false] was read and verified`. The emitter reads the flag only on a matmul, so such a kernel would run as if unflagged. The only artifact any command reads back is the one `compile` has just written.

**The 14 failures in suite run 10:**
- 1 is ADV-T1.
- 2 are the earlier `#[ignore]`d ADV-FM-1 and ADV-FM-2.
- 1 is `adversary_tiling_stale_artifact`, which needs `ADVERSARY_TILING_BASE_ARTIFACT` set.
- 10 are in 4 stand-in or old-device targets: "Unable to load create_debug_utils_messenger_ext", then "panic in a function that cannot unwind". Without `MODELSPEC_VK_VALIDATION` all 11 of their cases pass (4, 3, 3 and 1). This unit does not change those files. Finding F4.

**F3, bank conflicts (my arithmetic, not measured).** Transposed `a` is staged into the unpadded `a_stride = depth`. That gives 8-way conflicts on the one-row tile, 4-way on the two-row tile and 2-way on the wide tile. Transposed `b` on the register-blocked tiles has the unpadded `b_stride = cols`, which gives 8-way and 16-way. This goes against the emit.rs comment "takes few stores into one bank at a time", and it may explain the dW2 +18 % regression. Padding `a_stride` and `b_stride` for transposed staging is the fix to try.

**Could not break:** bit-exactness at partial tiles, the arithmetic order, the workgroup memory size, module-id separation, base artifacts reading at head, and device training against the CPU on 4 fold programs (for example prediction_transpose final loss: CPU 187.71346, GPU 187.71347).

Cleanup: heartbeat stopped by PID, scratch targets cleaned (718 MiB and 1005 MiB), lease ended (`session-end` exit 0).

```findings
- file: crates/modelspec-core/tests/adversary_transposed_artifact.rs
  line: 106
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: "ADV-T1: artifact::read and verify_plan accept a non-matmul kernel with transposed [true,false] (module id recomputed, artifact resealed); no per-kind check in WireKernel::kernel or verify.rs. No command loads such an artifact."
- file: crates/modelspec-core/src/schedule.rs
  line: 266
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "With the prediction exclusion (|| bound.contains(&node.id)) deleted, 92 core cases stay green and verify_plan accepts the plan; only adversary_transposed_fold.rs catches it."
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: null
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "My arithmetic, not measured: transposed staging (Staging::of, TRANSPOSED_A_RUN) stores into the unpadded a_stride=depth (8/4/2-way bank conflicts on the one-row/two-row/wide tiles) and b_stride=cols (8/16-way on the register-blocked tiles). This goes against the emit.rs comment and may explain dW2 +18 %. Try padding the strides for transposed staging."
- file: crates/modelspec-cli/tests/inspect_device_stand_in.rs
  line: null
  category: robustness
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: "With MODELSPEC_VK_VALIDATION=1, 10 cases in 4 stand-in/old-device targets panic in ash ('Unable to load create_debug_utils_messenger_ext', 'panic in a function that cannot unwind'); without it all 11 of their cases pass. This unit does not change those files."
```
