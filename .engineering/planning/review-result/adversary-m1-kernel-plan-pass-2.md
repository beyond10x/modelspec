---
format: aep.planning-md/3
id: review-result:adversary-m1-kernel-plan-pass-2
kind: review-result
status: active
title: Adversary pass 2, story:m1-kernel-plan
relations:
- reviews: story:m1-kernel-plan
revision: 1
---
unit: story:m1-kernel-plan, working tree of impl/m1-kernel-plan (base 3a62a7a, uncommitted) in modelspec-impl-kplan, as of 2026-10-22
verdict: INFEASIBLE (4 red verifier cases, none reached by the schedule) plus 1 CONFIRMED note reachable from a source file
cases: executed 685→690, red 4
origin: introduced 6 / pre-existing 0 / undecided 0
wrote-outside-worktree: 5 paths, all under the assigned scratch directory (part 6)
needs-coordinator: the spec says the step plan "ends with its only fence", but pass-1's green test `the_step_plan_recorded_twice_back_to_back_verifies` (and the implementor's `the_mean_loss_step_plan_recorded_twice_verifies`) require `verify_plan` to accept a plan with two fences. Both cannot hold. Decide whether `verify_plan` gets one fence per plan plus a separate back-to-back check, or whether the spec drops "only" and the verifier learns to tell a training-step boundary from a mid-plan fence.

## 1. Diff stat (what I touched)

`git --no-pager diff --stat 3a62a7a` is the implementor's diff, unchanged by this pass: 94 files, +1010 −112. This pass added one untracked test file and nothing else:

```
?? crates/modelspec-core/tests/adversary2_kplan.rs
```

No implementation file, spec, store or doc was edited.

## 2. Cases added (crates/modelspec-core/tests/adversary2_kplan.rs)

| Case | Asserts | Now |
|---|---|---|
| `pass_1_case_7_with_a_fence_appended_is_still_refused` | pass-1 case 7 unchanged, plus a final fence, is refused | RED |
| `a_step_reading_an_updated_parameter_as_its_second_operand_is_refused` | two_layer_relu with a Copy placed between the two SGD updates, reads `[w2, w1]`, barriers on both sides, is refused | RED |
| `a_prediction_view_no_plan_writes_is_refused` | `prediction_view` moved to a fresh buffer that no step writes is refused | RED |
| `a_loss_view_the_step_plan_never_writes_is_refused` | `loss_view` moved to a fresh 8-element buffer that no step writes is refused | RED |
| `every_plan_recorded_three_times_verifies` | the step and forward plans of linear, two_layer, mean-loss and returns-`w`, each recorded 3 times, verify | green |

Red output from the first run of this file alone (`b10x-build-slot cargo test -p modelspec-core --test adversary2_kplan`), verbatim:

```
test a_loss_view_the_step_plan_never_writes_is_refused ... FAILED
test a_prediction_view_no_plan_writes_is_refused ... FAILED
test pass_1_case_7_with_a_fence_appended_is_still_refused ... FAILED
test a_step_reading_an_updated_parameter_as_its_second_operand_is_refused ... FAILED
test every_plan_recorded_three_times_verifies ... ok
---- a_loss_view_the_step_plan_never_writes_is_refused stdout ----
panicked at crates/modelspec-core/tests/adversary2_kplan.rs:202:5:
the loss view is eight elements in buffer b11, which no step writes: Ok(())
---- a_prediction_view_no_plan_writes_is_refused stdout ----
panicked at crates/modelspec-core/tests/adversary2_kplan.rs:188:5:
the prediction view names buffer b11, which neither plan writes: Ok(())
---- pass_1_case_7_with_a_fence_appended_is_still_refused stdout ----
panicked at crates/modelspec-core/tests/adversary2_kplan.rs:118:5:
a step plan with two fences, whose second segment reads `w` after the update of the same training step, verifies: Ok(())
---- a_step_reading_an_updated_parameter_as_its_second_operand_is_refused stdout ----
panicked at crates/modelspec-core/tests/adversary2_kplan.rs:171:5:
step 27 reads parameter buffer b2 after update step 24 wrote it in the same recording: Ok(())
test result: FAILED. 1 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

## 3. Suite run (after the cases existed)

`CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot cargo test --workspace --no-fail-fast`. Exit 101; the full log is `scratch/adversary2/suite.log`.

```
test result: FAILED. 1 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
error: 1 target failed:
    `-p modelspec-core --test adversary2_kplan`
(sum of all test results: passed 686, failed 4)
```

The before count is 685, from the implementor's report. 685 + 5 = 690. All other targets are green.

## 4. Findings

1. **Pass-1 case 7 is not covered.** verify.rs:1038–1048 treats every fence as a recording boundary. ReadAfterUpdate (1210) and AllocateAfterUse (1108) only look inside the current segment. So case 7 is refused only because it lacks a final fence (PlanNotFenced). Add one fence and it verifies, even though a runner records the whole plan once per training step.
   - Contradicts: spec/domains/artifact.yaml:221 ("one training step … ends with its only fence") and DESIGN.md:164 ("each ends with its only fence").
   - What reaches it: nothing today. `record` emits exactly one fence (schedule.rs `record`), and no artifact loader exists yet (story:m1-compile-artifact). Verdict: INFEASIBLE, warning.
   - Proposed fix: refuse a fence that is not the last step, and check back-to-back recording with a separate function. That turns two existing green tests red, so it needs the coordinator.
2. **ReadAfterUpdate checks only the first parameter a step reads.** verify.rs:1205–1208 uses `.find(...)` for the first parameter-buffer read. A step that reads `[w2, w1]` after w1's update but before w2's passes.
   - What reaches it: nothing. No scheduled step reads two parameters after an update. Verdict: INFEASIBLE, note.
   - Proposed fix: check every parameter read, not just the first.
3. **No check that the bound views are written.** verify.rs:910 (`verify_plan`) never checks that the forward and step plans write `prediction_view`, or that the step plan writes `loss_view` as a 1-element view. The spec promises both (artifact.yaml:230, :233).
   - What reaches it: nothing. Pass-1's source test shows the schedule writes both. Verdict: INFEASIBLE, note.
4. **Back-to-back recording leaves the batch upload unspecified.** artifact.yaml:72 has a runner write each batch into `DataSpec.view` "before a step". artifact.yaml:221 lets a runner record the plan back to back into one command buffer. No plan step loads a batch, and the fence rule orders the next recording after the fence, not after any transfer the runner adds.
   - Effect: every recording reads the same batch, unless the runner adds its own transfer plus a barrier, and no contract says so.
   - What reaches it: no runner yet (M1 Vulkan stories). Verdict: NEEDS-CHANGE, note: one sentence in the spec.
5. **"No two values share a buffer" is false for in-place updates.** artifact.yaml:227 says it, and DESIGN.md:164 says "one buffer … per value". But the SGD update writes its parameter's buffer in place: in the golden, `%14 writes b2`, and b2 is `@LinearReg.w`'s view. Verdict: CONFIRMED, note, doc only.
6. **E-TARGET-010 does not refuse tensors no Vulkan device can bind.** The size limit is "more bytes than 64 bits count" (DESIGN.md § 8.1). `modelspec ir big_param.mspec --stage kernel` exits 0 for a `[4, 1073741824]` parameter:
   - it gets a 17,179,869,184-byte buffer b2;
   - k0 gets workgroups `[65535, 2049, 1]`, which is 8.6e9 invocations.
   
   `maxStorageBufferRange` is a `uint32_t` in Vulkan (from Vulkan spec knowledge, not checked here), and a u32 linear invocation index wraps above 2^32. So the source schedules but cannot run.
   - What reaches it: any checked source with a tensor over 4 GiB. The harm lands in the M1 runtime stories. Verdict: CONFIRMED, note.

## 5. Attacked, not broken

- **Fence soundness:** a verified plan ending in a fence stays race-free when recorded back to back, by argument (every no-dependency step in recording k+1 closes over fence k) and by case 5 (4 sources × 2 plans × 3 recordings).
- **Model match:** every barrier the verifier credits sits between the two steps in plan order, so a runner that records in order with full barriers is no weaker than the verifier's model.
- **Copy path:** the Copy for a parameter prediction is ordered before the update (UpdateMissesReader covers Copy readers). The forward plan's Copy writes no parameter. Copy hazards are checked.
- **Broadcast path:** a literal prediction gets a 1-workgroup broadcast in both plans and verifies.
- **Spread arithmetic:** the 65535³ workgroup boundary is correct.
- **Symbolic dims:** not reachable (`ir/tensor.rs:61`).

## 6. Paths written outside the worktree

- ~/.cache/modelspec-wave-m1-a/m1-kernel-plan/scratch/adversary2/case.log
- ~/.cache/modelspec-wave-m1-a/m1-kernel-plan/scratch/adversary2/suite.log
- ~/.cache/modelspec-wave-m1-a/m1-kernel-plan/scratch/adversary2/big_param.mspec
- ~/.cache/modelspec-wave-m1-a/m1-kernel-plan/scratch/adversary2/big.out
- ~/.cache/modelspec-wave-m1-a/m1-kernel-plan/scratch/adversary-2.md (this report)

## 7. Findings block

```findings
[
 {
  "file": "crates/modelspec-core/src/ir/verify.rs",
  "line": 1043,
  "category": "contract-drift",
  "severity": "warning",
  "verdict": "INFEASIBLE",
  "origin": "introduced",
  "message": "Every fence opens a new recording for ReadAfterUpdate and AllocateAfterUse, so pass-1 case 7 with a fence appended verifies, against the spec's \"ends with its only fence\" for one training step; the fix conflicts with the green back-to-back tests and needs the coordinator."
 },
 {
  "file": "crates/modelspec-core/src/ir/verify.rs",
  "line": 1205,
  "category": "mutant",
  "severity": "note",
  "verdict": "INFEASIBLE",
  "origin": "introduced",
  "message": "ReadAfterUpdate checks only the first parameter view a step reads, so a step reading [w2, w1] after w1's update and before w2's verifies."
 },
 {
  "file": "crates/modelspec-core/src/ir/verify.rs",
  "line": 910,
  "category": "contract-drift",
  "severity": "note",
  "verdict": "INFEASIBLE",
  "origin": "introduced",
  "message": "verify_plan accepts a prediction_view no plan writes and a loss_view of eight elements no step writes, against artifact.yaml:230 and :233."
 },
 {
  "file": "spec/domains/artifact.yaml",
  "line": 72,
  "category": "judgement",
  "severity": "note",
  "verdict": "NEEDS-CHANGE",
  "origin": "introduced",
  "message": "The contract lets a runner record the step plan back to back in one command buffer and also says the runner writes each batch before a step, but no plan step or ordering rule covers a per-step batch upload between recordings."
 },
 {
  "file": "spec/domains/artifact.yaml",
  "line": 227,
  "category": "contract-drift",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "\"No two values share a buffer in M1\" (and DESIGN.md:164 \"one buffer per value\") is false for the in-place SGD update, which writes the parameter's buffer (golden %14 writes b2)."
 },
 {
  "file": "crates/modelspec-core/src/schedule.rs",
  "line": 387,
  "category": "boundary",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "A checked source with a [4, 1073741824] parameter gets a 16 GiB buffer and 8.6e9 invocations from ir --stage kernel with exit 0, past Vulkan's u32 maxStorageBufferRange and a u32 invocation index, with no E-TARGET-010."
 }
]
```
