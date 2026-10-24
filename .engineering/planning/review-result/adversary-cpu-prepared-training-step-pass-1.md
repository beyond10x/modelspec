---
format: aep.planning-md/3
id: review-result:adversary-cpu-prepared-training-step-pass-1
kind: review-result
status: active
title: Adversary pass 1 on cpu-prepared-training-step
relations:
- reviews: story:cpu-prepared-training-step
revision: 1
---
Now the test file.
unit: story:cpu-prepared-training-step, uncommitted working tree on d2c44c7 (branch impl/cpu-prepared-training-step)
verdict: NEEDS-CHANGE
cases: executed 391→397, red 3
origin: introduced 1 / pre-existing 0 / undecided 0
wrote-outside-worktree: 2 paths
needs-coordinator: none

**1. Diff stat.** `git --no-pager diff --stat` shows only the implementor's 5 tracked files (784+, 213-). I edited none of them. My two files are new and untracked, so they show only in `git status --short`:
```
?? crates/modelspec-runtime/tests/adversary_prepared_plan.rs
?? crates/modelspec-runtime/tests/adversary_prepared_step.rs
```

**2. Cases added**

`crates/modelspec-runtime/tests/adversary_prepared_step.rs` builds a program where two parameters share one gradient value. Autodiff passes the adjoint of a same-shape `Add` to both operands unchanged (`crates/modelspec-core/src/ops.rs:208`), so the training program's outputs are `[loss, g, g, upd, upd]`.
- `a_shared_gradient_keeps_the_loss_and_both_gradients`: `sgd_step` should return the forward loss 14/3 and 2 gradients. **Red.**
- `a_prepared_step_with_a_shared_gradient_keeps_the_loss`: same check over 2 `PreparedStep` steps. **Red.**
- `a_run_with_a_shared_gradient_reports_the_forward_loss`: the `.mspec` source `return add(matmul(x, w), add(b1, b2));` through `runner::run`; `losses[0]` should equal the forward loss of the initial parameters. **Red.**
- `the_interpreter_outputs_of_a_shared_gradient_are_right`: the interpreter's output 0 is the true loss, and positions 1 and 2 hold the same gradient `[2/3, 4/3, 2]`. Green, which places the defect in `PreparedStep`, not in `Prepared`.

Red output of this file run alone (`b10x-build-slot cargo test -p modelspec-runtime --features vulkan --test adversary_prepared_step`). The first run failed only on typos in my own cases: a forward program without `output(loss)`, and `+` instead of `add(...)` in the source. After fixing those:
```
step 0: loss 0.6666667 against the forward loss 4.6666665   left: 1059760811 right: 1083528533
loss 0.6666667 against the forward loss 4.6666665            left: 1059760811 right: 1083528533
loss of step 0 1.0391272 against the forward loss 3.8919926  left: 1065681439 right: 1081677416
test result: FAILED. 1 passed; 3 failed
```

`crates/modelspec-runtime/tests/adversary_prepared_plan.rs` has 2 cases, both green:
- An output that is an input, an output listed twice, an output that reuses a freed intermediate's slot, an in-place transpose whose source is an output, and a parameter as output. Three runs with every output recycled, compared with `evaluate_all` and literals; the feed stays untouched.
- `recycle` given a tensor of the right element count but another shape.

**3. Suite runs** (after the cases existed)
| command | exit | result |
|---|---|---|
| `b10x-build-slot cargo test -p modelspec-runtime --features vulkan --no-fail-fast` | 101 | 394 passed, 3 failed (only the 3 above), 53 ignored |
| `b10x-build-slot cargo test -p modelspec-core --no-fail-fast` | 0 | 356 passed, 1 ignored |

`executed 391` = 397 minus my 6 cases. I derived it and did not run the suite with my files deselected.

**4. Finding**
- **What was measured:** `crates/modelspec-runtime/src/train/optimizer.rs:211-216`. `roles` defaults every output position to `Role::Loss`, and gradients are placed with `position(value)`, which returns the first listing only. With a value listed twice, the later position keeps `Role::Loss`, so `:281-283` overwrites the loss with `g[0]`. `self.gradients` also keeps only one of the two parameters.
- **Why it is introduced:** the base `evaluate_step` read `outputs[0]` and cloned an output used twice (`git show 93121b3:crates/modelspec-runtime/src/train/optimizer.rs`, lines 163-180). I did not run the base itself; the green interpreter case shows that the base path would have returned the right loss.
- **What reaches it:** `modelspec train --backend cpu` → `crates/modelspec-cli/src/train.rs:727` → `run_from` → `PreparedStep::evaluate` (`runner.rs:217-231`), on any model that adds two same-shape parameters.
- **Effects:**
  - Printed losses are wrong, which breaks the `same-bits` acceptance.
  - The observer is missing one parameter's gradient.
  - The non-finite check reads the wrong value.
  - Parameter updates are still correct, because they use the internal `g`.
- **Fix, not applied:** set the loss from position 0, and fill roles per listing: each gradient or update takes its own position, or gets a clone when listings share one, as the base did.

**5. Attacked and not broken**
- A free before a read: a node never takes a slot freed by its own operands (`interpreter.rs:239-255`).
- In-place transpose with a live source, or a source that is an output: green.
- Output that is an input or a parameter: copied, the feed is not moved.
- Shapes differing between steps: shapes are static (`static_shape`, `:655`), and eval uses its own `evaluate`.
- Resume: the existing `checkpoint.rs` case `a_resumed_run_equals_one_that_never_stopped` passes.
- Run of only evaluation steps: never prepares.
- `evaluate_all` after a `Prepared` run: it builds its own plan.
- Per-thread counters: no timing assertion and no cross-thread use in the tests.

**6. Paths written outside the worktree**
- `<wave-root>/prepared/scratch/adversary/runtime-suite.log`
- `<wave-root>/prepared/scratch/adversary/core-suite.log`

```findings
- file: crates/modelspec-runtime/src/train/optimizer.rs
  line: 211
  category: acceptance
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: when two parameters share a gradient value the outputs list it twice; the second listing keeps the default Role::Loss, so PreparedStep (and sgd_step, run_from, modelspec train) reports g[0] as the loss and drops one parameter's gradient, where the base read outputs[0] and cloned shared outputs
```
