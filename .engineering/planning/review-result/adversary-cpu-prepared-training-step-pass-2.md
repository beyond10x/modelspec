---
format: aep.planning-md/3
id: review-result:adversary-cpu-prepared-training-step-pass-2
kind: review-result
status: active
title: Adversary pass 2 on cpu-prepared-training-step
relations:
- reviews: story:cpu-prepared-training-step
revision: 1
---
```
unit: story:cpu-prepared-training-step, uncommitted working tree on d2c44c7 (branch impl/cpu-prepared-training-step)
verdict: NEEDS-CHANGE
cases: executed 398→403, red 1
origin: introduced 1 / pre-existing 0 / undecided 0
wrote-outside-worktree: 4 paths
needs-coordinator: none
```

The pass 1 blocker is fixed. One acceptance case is still red, and it is a warning, not a blocker: when one gradient is listed three times, step 2 makes 10 allocations for 9 outputs, so `storage-reused` fails on that program. Loss, gradients, updates, the non-finite check and resume are all correct on shared-gradient programs.

**1. `git --no-pager diff --stat`**
The tracked diff holds only the implementor's 5 files (798+, 213-). I edited none of them. My one file is new and untracked:
`?? crates/modelspec-runtime/tests/adversary_prepared_step_2.rs`

**2. Cases added** in `crates/modelspec-runtime/tests/adversary_prepared_step_2.rs`. It has its own counting global allocator.

| case | asserts | now |
|---|---|---|
| `a_second_step_with_a_gradient_listed_three_times_allocates_at_most_one_per_output` (:107) | `.mspec` model `add(matmul(x, w), add(add(b1, b2), b3))`. Outputs `[loss, gw, g, g, g, uw, u1, u2, u3]` (9). Step 2 allocates at most 9 times | **red** |
| `a_gradient_listed_three_times_keeps_the_loss_gradients_and_updates` (:142) | 3 steps. The loss matches the forward loss bit for bit. Each of the 3 gradients and each update matches the `evaluate_all` value (`Keep::Every` path) | green |
| `a_parameter_the_loss_does_not_read_keeps_its_value_beside_a_shared_gradient` (:219) | The parameter the loss does not read keeps its value and version 0. The others equal `sgd_step` | green |
| `a_run_with_a_shared_gradient_stops_on_an_infinite_loss_at_step_0` (:273) | Biases set to 1e20: the loss is `inf` and the gradients are finite. `run_with` stops at step 0 with "the loss of step 0 is not finite". The observer sees loss `inf` and 4 gradients, the 3 bias gradients bit-identical. Under the pass 1 defect the run would not have stopped | green |
| `a_resumed_run_with_a_shared_gradient_equals_one_that_never_stopped` (:319) | 2 steps then 3 resumed, against 5 straight: losses equal bit for bit, final state equal | green |

The first run failed to compile on a typo of mine (`&&str` passed to `declare_param`). Red output of the second run, the file alone (`b10x-build-slot cargo test -p modelspec-runtime --features vulkan --test adversary_prepared_step_2`, exit 101):
```
the second step allocated 10 times; it returns 9 outputs
test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
```

**3. Suite runs** (after the cases existed)

| command | exit | result |
|---|---|---|
| `b10x-build-slot cargo test -p modelspec-runtime --features vulkan --no-fail-fast` | 101 | 402 passed, 1 failed (only the red case above), 53 ignored |
| `b10x-build-slot cargo test -p modelspec-core` | 0 | 356 passed, 1 ignored |
| `b10x-build-slot cargo clippy -p modelspec-runtime --features vulkan --all-targets -- -D warnings` | 0 | |
| `cargo fmt --check` (after `rustfmt` on my file) | 0 | |

`executed 398` is the implementor's gate count; 403 is 398 plus my 5 cases.

**4. Finding**
- **What was measured:** `crates/modelspec-runtime/src/cpu/interpreter.rs:453` clones each output's shape (`self.plan.shapes[node].clone()`) on every run, and `:448` allocates data for every copy. Step 2 therefore makes (outputs − 1) shape allocations (the rank-0 loss shape is free) plus one per extra listing. That is 8 + 2 = 10 against a bound of 9.
- **Why the existing test passes:** with two listings the count lands exactly on the bound (4 + 1 = 5). The doc comment of `prepared_step.rs:361` says as much: "one allocation per step for its data beside one per output shape". With three or more listings, or two different shared values, the acceptance line `storage-reused` fails.
- **What reaches it:** any `.mspec` model that adds three same-shape parameters, through `modelspec train --backend cpu` → `run_from` → `PreparedStep::evaluate`.
- **Effect:** correctness is unaffected. The cost is a few small allocations per step.
- **Fix, not applied:** keep the shape with the slot's storage, or reuse the shape of the recycled tensor, so a moved output allocates nothing and only a copy allocates.
- **Origin:** introduced. The acceptance line and the code belong to this unit.

**5. Attacked and not broken**
- **Position rule:** `verify_step_writes` → `check_outputs` (`verify.rs:579-581`) requires outputs `[loss, gradients in ParamId order, updates in ParamId order]`, and update keys equal gradient keys (`:614-627`). The `1+k` / `1+G+k` positions are exact for every verified step, so the `debug_assert`s can be dropped in release.
- **Update shared with a gradient or the loss:** the verifier refuses it, since updates must be tagged `Optimizer` and gradients and the loss cannot be (`:589-598`, `:637-643`).
- **Two parameters with one update value:** `attach_sgd` builds one sub per parameter. Nothing reaches it.
- **Zero parameters:** writes are empty and the outputs are `[loss]`. I wrote no case for it.
- **Copy on earlier listings, moved buffer on the last:** bits equal `evaluate_all`. The moved buffer is recycled at the last position only (`plan.moves`, `interpreter.rs:265-273`).
- **`evaluate_step`:** it was not removed but reimplemented on `PreparedStep`. Callers in the CLI tests, `dense_batch_reuse.rs` and the bench still compile and pass. No stale path.

**6. Paths written outside the worktree** (all under `<wave-root>/prepared/scratch/adversary/`)
- `pass2-red.log`
- `pass2-runtime-suite.log`
- `pass2-core-suite.log`
- `pass2-clippy.log`

```findings
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 453
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: every output clones its shape each run and every extra listing copies its data, so a program listing one gradient three times (add(add(b1, b2), b3), reachable from .mspec) allocates 10 times in step 2 against 9 outputs, failing storage-reused; tests/adversary_prepared_step_2.rs:133 is red
```
