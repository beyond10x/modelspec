---
format: aep.planning-md/3
id: review-result:adversary-dropout-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-dropout
relations:
- reviews: story:bert-dropout
revision: 1
---
unit: story:bert-dropout, commit 5fcaf12 on impl/bert-dropout, worktree wt-ed2ada376573 plus my two untracked test files
verdict: findings 5 (blocker 0, warning 1, note 4)
cases: executed 1687→1695, red 2
origin: introduced 5 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 path (the assigned scratch directory `<wave-root>/dropout/adv1-scratch/`)
needs-coordinator: ADV-DROP-1. D47 names a site `<Model>.<fn>.dropout#<k>`. The unit also wrote "a function inlined twice keeps one site" into DESIGN.md § 2.3. Keep that rule or change it before modules arrive (DESIGN § 11 iteration 3)?

**1. `git --no-pager diff --stat`**: empty. I changed no tracked file. `git status --short` shows only my two new test files:
```
?? crates/modelspec-cli/tests/adversary_dropout_cli.rs
?? crates/modelspec-runtime/tests/adversary_dropout_numerics.rs
```

**2. Cases added**

| file | case | asserts | now |
|---|---|---|---|
| runtime `adversary_dropout_numerics.rs` | `adv_drop_1_function_inlined_twice_draws_two_masks` (`#[ignore]`) | the 2 `Dropout` nodes of `block(block(forward(x)))` read different tokens | red |
| runtime | `adv_drop_boundary_u_equal_p_keeps_the_element` | `p` set to an element's exact uniform; that element is kept in `Dropout` and `DropoutGrad` | green, kills mutant a |
| runtime | `adv_drop_id_takes_the_step_as_8_bytes` | step `2^32+1` matches a hand blake3 reference and differs from step 1 | green |
| runtime | `adv_drop_dropped_element_is_positive_zero_for_negative_values` | dropped elements are `+0.0` bits for negative x and negative g | green, kills mutant i |
| runtime | `adv_drop_f64_reference_multiplies_by_the_widened_f32_scale` | at `p = 0.1` the f64 kept value is `x * f64::from(1f32/(1f32-p))`, bit for bit | green, kills mutant j |
| runtime | `adv_drop_runner_keys_each_step_with_the_global_step_token` | `run_from` at start 0, 1 and 7: the recorded loss equals the step program run with `token(site, start)`, bit for bit | green, kills e, e1, e2 |
| runtime | `adv_drop_vulkan_artifact_is_e_target_011_at_the_call` | `compile_artifact(.., Vulkan)` gives one E-TARGET-011 at 7:17 | green |
| cli `adversary_dropout_cli.rs` | `adv_drop_1_ir_prints_one_site_per_inlined_call` (`#[ignore]`) | `ir --stage tensor` shows 2 different sites | red |
| cli | `adv_drop_checker_codes_of_calls_the_unit_does_not_try` | 7 calls: positional `p` gives E-TYPE-001, `p` before the operand E-TYPE-002, an unknown `q` E-TYPE-002, `1e-400` ok, `-0.0` ok, `1e39` E-TYPE-022, `0.99999994` ok | green |
| cli | `adv_drop_resume_draws_the_masks_of_the_global_steps` | 12 steps in one run equal 5 steps, a checkpoint, then 7 resumed steps (param lines and the step-12 loss line) | green, kills e2 |

Red output of each red case, run alone before the suite:
```
---- adv_drop_1_function_inlined_twice_draws_two_masks stdout ----
assertion `left != right` failed: both calls of `block` read one token, sites ["Reg.block.dropout#0"]
  left: ValueId(4)
 right: ValueId(4)
```
```
---- adv_drop_1_ir_prints_one_site_per_inlined_call stdout ----
%4: Tensor<i32, [2]> = Input "Reg.block.dropout#0"
%5: Tensor<f32, [8, 3]> = Dropout [p = 0.5, site = Reg.block.dropout#0] %3, %4
%6: Tensor<f32, [8, 3]> = Dropout [p = 0.5, site = Reg.block.dropout#0] %5, %4
  left: "Reg.block.dropout#0"
 right: "Reg.block.dropout#0"
```

**3. Suite runs** (`cargo test --workspace --no-fail-fast` in the worktree, after the cases existed)
- With my cases: exit 0, 1695 passed, 0 failed, 143 ignored.
- Same run with `-- --skip adv_drop_`: exit 0, 1687 passed, 141 ignored.
- With `--ignored`, both ADV-DROP-1 cases fail as quoted above.

**Mutants.** Each was planted in a scratch copy and run against the unit's suite without my files. "Unit suite" here means core and runtime `--lib`, `gradients_dropout`, `train_dropout` and `check_dropout` (188 cases). For the survivors, mutants a, e2, i and j were then planted together and the full workspace ran: 1687 passed, 0 failed.

| mutant | unit suite | my files |
|---|---|---|
| a: `u > p` instead of `>=` | survived | killed (boundary case) |
| b: stream tag 3 | killed (`dropout_is_stream_tag_4…`, `dropout_mask_derivation…`, `dropout_grad_is_the_scaled_mask…`) | killed |
| c: step as 4 bytes | killed (`a_dropout_token_is_the_draw_id…`, `dropout_mask_derivation…`) | killed |
| d: `x / (1 - p)` per element | killed (`dropout_mask_derivation…`) | killed |
| e: token of step `s-1` (wrapping) | survived | killed (runner case) |
| e1: token of step `s+1` | survived | killed (runner case) |
| e2: token of `step - start` | survived | killed (runner and resume cases) |
| f: `DropoutGrad` without the scale | killed (`dropout_grad_is_the_scaled_mask…`, `gradients_dropout_match…`) | killed |
| g: every site `#0` | killed (`lower::tests::dropout_sites_are_named…`, `dropout_sites_and_steps_differ`) | survived |
| h: dropout kept in the forward program | killed (`train_dropout` ×2) | killed |
| i: dropped element `v * 0` (gives `-0.0`) | survived | killed |
| j: scale computed in the element type (f64 scale in f64 runs) | survived | killed |
| k: Philox word 1 instead of word 0 | killed (`dropout_mask_derivation…`, `dropout_grad…`) | killed |

**4. Findings** (all on 5fcaf12)

- **ADV-DROP-1, note, CONFIRMED, introduced.** `crates/modelspec-core/src/lower.rs:342`
  - What was measured: when a model function is inlined twice, both copies share one token, so both draw the same mask. In `dropout(dropout(h))`, kept elements are scaled by 4 with probability 0.5, so the expected value is 2h instead of h.
  - What reaches it: only the train block's loss calling a helper function twice. A model function cannot call another (E-NAME-007), and `forward` runs exactly once (E-TYPE-017).
  - Planned work does not reach it: BERT-tiny writes its layers out inside `forward` (story:bert-tiny-cpu-model). It becomes real once modules let one function run per layer.
  - It is documented on purpose in DESIGN.md § 2.3, so changing it is the coordinator's call.
- **ADV-DROP-2, warning, CONFIRMED, introduced.** `crates/modelspec-runtime/src/train/runner.rs:233`
  - Which step's token the runner passes is untested: mutants e, e1 and e2 pass the full workspace.
  - e2 (counting steps from the resume point instead of the global step) silently breaks resume reproducibility.
  - story:bert-adamw-warmup edits `runner.rs` next.
  - Fix: keep my two green cases as the unit's tests.
- **ADV-DROP-3, note, CONFIRMED, introduced.** `crates/modelspec-runtime/src/train/init.rs:155`
  - What was measured: the `u >= p` boundary of D47 is unpinned (mutant a survives).
  - What reaches it: only literals below `2^-9` that equal an element's uniform exactly.
- **ADV-DROP-4, note, CONFIRMED, introduced.** `crates/modelspec-runtime/src/cpu/interpreter.rs:762`
  - What was measured: "a dropped one is `+0.0`" (DESIGN § 2.3) is unpinned, because every unit test feeds positive x and g (mutant i survives).
- **ADV-DROP-5, note, CONFIRMED, introduced.** `crates/modelspec-runtime/src/cpu/interpreter.rs:757`
  - What was measured: the f64 reference using the widened f32 scale (DESIGN § 2.3) is unpinned (mutant j survives).

**5. Attacked and not broken**
- The mask derivation: blake3 with the step as 8 LE bytes, Philox counter `[i, 4, lo, hi]`, word 0, the § 2.3 uniform.
- Tag 4 between `Masking` (3) and `DataOrder` (5).
- Scale computed once in f32; the gradient uses the same mask and the same scale bits.
- The token is the global step, also after `--resume`.
- The final loss line, `--eval` and `infer` all use the forward program without dropout.
- E-TARGET-011 lands at the call.
- Checker codes for 7 more call shapes.
- No collision between token names and user names: identifiers cannot contain `.` or `#`.
- The CPU artifact has no plan; its token takes an unnamed 8-byte buffer. The Vulkan backend refuses dropout, so nothing reaches that buffer today.
- DESIGN.md § 2.3, § 4 and § 8, `diagnostics.md` (E-TYPE-022), `operators.md` and the CHANGELOG line agree with the code.

**6. Paths written outside the worktree.** Everything is inside the assigned `<wave-root>/dropout/adv1-scratch/`:
- `mut/`: a source copy of the tree, 20 MB. Its `target/` was removed with `cargo clean --target-dir` (9.4 GiB).
- `orig/`, `adv-tests/`, `logs/` and `probe-artifact/`.
- `unit-suite.sh`, `adv-suite.sh` and `mutate.sh`.

Inside the worktree, the CLI cases write `.mspec` and `.msckpt` files under `target/tmp/adversary_dropout_cli/`. The lease is ended.

```findings
- file: crates/modelspec-core/src/lower.rs
  line: 342
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: a model function inlined twice reuses one site token, so both calls draw the same mask and dropout(dropout(h)) has expectation 2h; reachable today only by calling a helper twice in the train block's loss
- file: crates/modelspec-runtime/src/train/runner.rs
  line: 233
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: no unit test checks which step's token the runner passes; tokens of step-1, step+1 or step-start pass the whole workspace suite, and step-start breaks resume reproducibility
- file: crates/modelspec-runtime/src/train/init.rs
  line: 155
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the D47 keep boundary u >= p is unpinned; the mutant u > p passes the whole workspace suite
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 762
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the +0.0 of a dropped element is unpinned because every unit test feeds positive values; v * 0 giving -0.0 passes the whole workspace suite
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 757
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the f64 reference using the widened f32 scale is unpinned; a scale computed in f64 passes the whole workspace suite
```
