---
format: aep.planning-md/3
id: review-result:adversary-dense-step-reuses-batch-buffers-pass-1
kind: review-result
status: active
title: Adversary pass 1 on dense-step-reuses-batch-buffers
relations:
- reviews: story:dense-step-reuses-batch-buffers
revision: 1
---
```
unit: story:dense-step-reuses-batch-buffers, uncommitted tree on base 93121b3 (modelspec-impl-dense-2)
verdict: CONFIRMED (1 warning, 2 notes)
cases: executed 367→375, red 2
origin: introduced 3 / pre-existing 0 / undecided 0
wrote-outside-worktree: 2 paths
needs-coordinator: decide whether a tensor's shape Vec counts as "an x allocation"; the fix touches cpu/tensor.rs, which is not in the unit's file assignment
```

**1. `git --no-pager diff --stat`**
```
 crates/modelspec-runtime/src/train/data.rs   | 87 +++++++++++++++++++---------
 crates/modelspec-runtime/src/train/runner.rs | 42 +++++++++-----
```
These are the implementor's numbers, unchanged. My only file is new and untracked: `crates/modelspec-runtime/tests/adversary_dense_batch_reuse.rs` (347 lines, 8 cases). No implementation file was touched. `cargo fmt --check` and clippy with `-D warnings` pass on it.

**2. Cases added (each red case was run alone first, captured verbatim)**

| Case | Asserts | Now |
|---|---|---|
| `a_refill_allocates_no_shape_of_x_or_y` | a fill after the first allocates no shape `Vec` | red |
| `a_batch_refilled_by_other_data_holds_only_its_declarations` | after a fill by data B, the batch equals `B.step(k)` | red |
| `a_step_across_the_epoch_boundary_holds_the_wrapped_rows` | N=6, B=4: rows 4,5,0,1 (literal values) | green |
| `a_step_past_the_end_of_the_order_wraps` | step 7 reads order positions 14 and 15→0 | green |
| `a_train_batch_then_an_eval_batch_holds_the_eval_rows` | train fill, then eval fill on a different sample set | green |
| `a_shorter_fill_leaves_no_stale_value` | B=4, then B=2, then B=4, same names | green |
| `the_default_batch_is_empty` | `default().inputs()` is empty | green |
| `the_per_step_allocation_count_is_deterministic` | 5 repetitions, synthetic first and dense first | green |

Red output:
```
step 1 allocated the shape of label (2 allocations in the step)
```
```
  left: {"label": .., "pixels": .., "x": HostTensor { shape: [2, 3], data: [1.0, 1.0, 0.0, 2.0, 0.0, 1.0] }, "y": ..}
 right: {"x": HostTensor { shape: [2, 3], data: [1.0, 1.0, 0.0, 2.0, 0.0, 1.0] }, "y": ..}
```

**3. Suite run**
`b10x-build-slot cargo test -p modelspec-runtime --features vulkan --no-fail-fast` exited 101: 373 passed, 2 failed (both mine, at `adversary_dense_batch_reuse.rs:209` and `:227`), 50 ignored. The 367 "before" count is my own derivation from that same run: 375 minus my 8 cases. The implementor's `dense_batch_reuse.rs` passed 6/6 on 5 repeated runs.

**4. Findings (tree as above)**

- **data.rs:270, warning, CONFIRMED, introduced (category: acceptance).**
  - **What was measured:** `HostTensor::from_parts(t.shape.clone(), data)` allocates a new shape `Vec` for each declaration on every fill. The red case is at `adversary_dense_batch_reuse.rs:209`.
  - **Your call:** the acceptance says "allocates no `x`, `y` or key `String`". The implementor reads "x" as the data buffer only. Their runner bound `dense - synthetic <= 2.0` (`dense_batch_reuse.rs:262`) is exactly these 2 shape allocations: I measured 106.125 vs 104.125 per step.
  - **What reaches it:** `run_from` → `Feed::inputs` → `fill_step`, once per step.
  - **Fix:** `HostTensor` has no way to hand back its shape (fields private, only `into_data`), so the fix needs something like a `pub(crate) fn data_mut(&mut self) -> &mut Vec<T>` in `cpu/tensor.rs`. That file is outside the assignment, so it needs a patch.
  - **Origin:** I called it `introduced` because the case cannot compile at the base (`fill_step` is new), but base `gather` also cloned the shape every step.
- **data.rs:257, note, INFEASIBLE, introduced (category: contract-drift).**
  - **What was measured:** `fill` only replaces entries for its own declarations, so keys left by another `DenseData` stay in the batch.
  - **What reaches it:** nothing found. Every batch in the runner is local to one data, and train and eval data of one program share names and B, so mixing them is correct (green case above).
  - **Impact:** documented as a precondition at data.rs:181. If it were ever hit, the interpreter would refuse the extra input (`cpu/interpreter.rs:37`) rather than compute with stale values.
- **dense_batch_reuse.rs:296, note, CONFIRMED, introduced (category: mutant).**
  - **What was measured:** `same_loss_bits_as_a_fresh_map_per_step` compares `run_with` against `data.step(k)`. `step` now delegates to `fill_step`, so the test cannot catch a sample-selection defect in `fill`. The `fresh` content checks in `assert_reused` have the same problem.
  - **What already covers it:** my literal-row cases above. The CLI tests with hand-computed losses (`modelspec-cli/tests/train_dense_data.rs:251`) also guard it, but I did not run them.

**5. Attacked, could not break**
- Batch size changing at the last step of an epoch: B is fixed per `DenseData` and the index wraps modulo N, so the shape never changes.
- Reuse across epochs with a shuffle order: green with literal values, including past the end of the order.
- Declaration count differing between fills: `tensors` is always `[DenseTensor; 2]`.
- Eval batch after a train batch: green.
- Stale value after a shorter fill: `data.clear()` handles it, green.
- `DenseBatch` default and `inputs()`: green.
- Allocation-count determinism: identical over 5 repetitions in either order. The runtime has no threads or lazy statics. The `<= 2.0` bound sits exactly on the measured value, so it is tight but deterministic, not flaky.

**6. Written outside the worktree**
- `<wave-root>/dense/scratch/adversary/suite.log`
- `<wave-root>/dense/scratch/adversary/suite-nff.log`

Fixtures went under the worktree's own `target/tmp/adversary_dense_batch_reuse/`. Session lease `adv-dense` was started and ended.

```findings
- file: crates/modelspec-runtime/src/train/data.rs
  line: 270
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: every fill after the first allocates a new shape Vec for x and y, so no-per-step-allocation holds only if a tensor's shape is not counted as part of x; the fix needs a refill accessor in cpu/tensor.rs, outside the unit's assignment
- file: crates/modelspec-runtime/src/train/data.rs
  line: 257
  category: contract-drift
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a public DenseBatch refilled by a DenseData with other declaration names keeps the previous data's entries; only the doc line at data.rs:181 forbids it and no caller does it
- file: crates/modelspec-runtime/tests/dense_batch_reuse.rs
  line: 296
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the same-loss-bits and assert_reused content checks compare against DenseData::step, which now delegates to fill_step, so they cannot detect a sample-selection defect in fill
```
