---
format: aep.planning-md/3
id: review-result:adversary-dense-step-reuses-batch-buffers-pass-2
kind: review-result
status: active
title: Adversary pass 2 on dense-step-reuses-batch-buffers
relations:
- reviews: story:dense-step-reuses-batch-buffers
revision: 1
---
```
unit: story:dense-step-reuses-batch-buffers, uncommitted tree on base 93121b3 (modelspec-impl-dense-2)
verdict: nothing found
cases: executed 376→380, red 0
origin: introduced 0 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 path
needs-coordinator: none
```

I found no defect in the correction. All 4 new cases pass, and the two edits the implementor made to the pass-1 file are honest.

**1. `git --no-pager diff --stat`**
```
 crates/modelspec-runtime/src/cpu/tensor.rs   | 12 ++++
 crates/modelspec-runtime/src/train/data.rs   | 88 ++++++++++++++++++++--------
 crates/modelspec-runtime/src/train/runner.rs | 42 ++++++++-----
```
These are the implementor's changes. My only file is new and untracked: `crates/modelspec-runtime/tests/adversary_dense_batch_reuse_2.rs`. I touched no implementation file.

**2. New cases (all green, so there is no red output to quote)**

| Case | What it asserts | Now |
|---|---|---|
| `a_refill_after_the_first_allocates_nothing` | after the first fill, a fill makes 0 allocations of any kind; checked on 9 steps without an order, 9 with one, and 3 eval batches; the first fill makes at least 4 | green |
| `the_last_step_of_an_uneven_epoch_wraps_to_sample_0` | N=5, B=2: step 2 holds samples 4 and 0, as literal values | green |
| `a_refill_to_the_same_count_in_other_dims_takes_the_new_shape` | the same names refilled as x [2,3] → [3,2] → [2,3] and y [2,2] → [3,2] → [2,2]: each fill holds the new shape and its rows, as literal values | green |
| `a_resumed_dense_run_gives_the_full_runs_losses` | `run_from` at start 3 gives losses bit-identical to the full run's `losses[3..]`, with and without an order | green |

**3. Suite run**
`b10x-build-slot cargo test -p modelspec-runtime --features vulkan --no-fail-fast` exited 0: 380 passed, 0 failed, 50 ignored. The before count of 376 is that same run minus my 4 cases. `adversary_dense_batch_reuse.rs` ran 8/8, `dense_batch_reuse.rs` 7/7, mine 4/4. `cargo fmt --check` and clippy with `-D warnings` are clean.

**4. Findings**
None. The tree is the uncommitted working tree on 93121b3.

**5. Attacked, could not break**
- **Edit 1** (`adversary_dense_batch_reuse.rs:223`): it asserts what data.rs:181 documents, that entries from the previous data remain. The expected values come from `step` rather than literal rows, but other literal-row cases cover the content. Its message names the doc comment rather than a line number, which is the only deviation from the correction's wording.
- **Edit 2** (`:355`): the difference `dense - synthetic == 0.0` is computed exactly on whole counts divided by 8, so it is not a rounding artefact. The runtime has no threads and no lazy statics (grep of `src/`), and the allocator only counts the recording thread. It would fail by 2 if `refill` allocated a shape again, so it is deterministic and not flaky.
- **`refill` with the same element count in other dims**: the shape is updated and the data length is right (case above). A change of rank cannot be reached: the language has no reshape, and the batch dimension leads every declaration.
- **Shrink then grow**: no stale tail is read, because `data.clear()` runs on every fill (pass-1 case `a_shorter_fill_leaves_no_stale_value`).
- **The declaration set changing size**: not reachable. `tensors` is `[DenseTensor; 2]`, and under the precondition the names never change.
- **`get_mut` on a missing key**: the first fill and eval batches are covered. `evaluate_dense` keeps its own batch, and eval data shares B with the training config.
- **N not divisible by B, and the wrap-around**: literal rows at N=5/B=2 and N=6/B=4, and past the end of an order.
- **Overflow, or a shape that does not match the data length, in `refill`**: not reachable. `refill` is `pub(crate)` with one caller, `fill`. That caller passes `t.shape`, whose element count `zeros` already checked with `checked_mul`, and its data length is `batch·sample`, which equals the shape's element count.
- **Other per-step callers of `DenseData::step` or `eval_batch`**: none. The Vulkan backend does not accept `--data`.

**6. Written outside the worktree**
- `<wave-root>/dense/scratch/adversary/suite-pass2.log`

The fixtures went under the worktree's own `target/tmp/adversary_dense_batch_reuse_2/`. I started and ended the lease `adv-dense`.

```findings
[]
```
