---
format: aep.planning-md/3
id: review-result:adversary-m1-kernel-plan-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-kernel-plan
relations:
- reviews: story:m1-kernel-plan
revision: 1
---
# Adversary pass 1 on story:m1-kernel-plan

Verdict: NEEDS-CHANGE; 14 cases added, 11 red (`cargo test --workspace --no-fail-fast`: 663 passed,
11 failed, all in `adversary_kplan_{verifier,sources,cli}.rs`). `core-no-vulkan`, `check-examples`,
`docs-check`, `schema-check` exit 0. Sent back to the implementor with all 11 to make green.

Attacked without a break: the ten refusals of acceptance 4 plus a missing barrier, allocate order,
a reduce in the forward, prediction and loss value ids, the linear golden (Mse scale 0.25 = 2/8),
`schema_hash` inputs, schedule determinism (no HashMap), `core-no-vulkan`, zero-size dimensions.

```findings
- file: spec/domains/artifact.yaml
  line: 211
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The contract lets a runner record the step plan back to back, but verify_plan refuses two back-to-back recordings (UpdateMissesReader update 27, reader 0) and no document requires a barrier between them.
- file: crates/modelspec-core/src/schedule.rs
  line: 264
  category: boundary
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: A forward returning a literal makes the prediction an immediate with no view, so ir --stage kernel reports E-INTERNAL-002 for a file that check accepts and train --backend cpu trains.
- file: crates/modelspec-core/src/schedule.rs
  line: 144
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: A forward returning a parameter (or an input) makes prediction_view that parameter's in-place view, and no plan writes it, against the spec summary.
- file: crates/modelspec-core/src/ir/verify.rs
  line: 941
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: verify_plan accepts bound views that alias a parameter buffer and views whose element count disagrees with the declared type.
- file: crates/modelspec-core/src/ir/verify.rs
  line: 992
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: A Dispatch's workgroups are not checked against its kernel's, so a single-invocation loss sum dispatched over [2,1,1] verifies.
- file: crates/modelspec-core/src/ir/verify.rs
  line: 1114
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: Hazards are checked only between Dispatch steps, so a Copy that writes a buffer a later dispatch reads without a barrier verifies.
- file: crates/modelspec-core/src/ir/verify.rs
  line: 1096
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: A dispatch reading a parameter after its in-place update, so reading version n+1 inside step n, verifies.
- file: crates/modelspec-core/src/schedule.rs
  line: 316
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: A checked source whose shape exceeds three dimensions of 65535 workgroups is reported as compiler bug E-INTERNAL-002 instead of a user-facing size limit.
- file: crates/modelspec-core/src/ir/kernel.rs
  line: 237
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: ExecutionPlan.buffers, PlanStep.value and ParameterSpec.param are not typed in spec/domains/artifact.yaml, so a loader has no typed buffer table.
- file: crates/modelspec-cli/tests/ir_tensor.rs
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: This file and 77 regenerated observatory schemas are outside the story's scope list; both changes are forced (schema-check, and the old unknown-stage test).
```
