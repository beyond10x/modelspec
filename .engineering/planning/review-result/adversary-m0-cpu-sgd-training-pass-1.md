---
format: aep.planning-md/3
id: review-result:adversary-m0-cpu-sgd-training-pass-1
kind: review-result
status: active
title: Adversary pass 1 (phase A) on story:m0-cpu-sgd-training
relations:
- reviews: story:m0-cpu-sgd-training
revision: 1
---
unit: story:m0-cpu-sgd-training, pass 1 phase A, uncommitted working tree on aa020fa (`impl/m0-cpu-sgd-training`)
verdict: NEEDS-CHANGE
cases: executed 417→425, red 4
origin: introduced 8 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory, 14 entries (part 6)
needs-coordinator: AGENTS.md:14 is your file (F7). Decide whether F3 is fixed in the doc or by a clamp, because that decides the assertion in my case. `task ci` stopped at the first red test binary, so `core-no-vulkan` and `check-examples` did not run under it. `no_vulkan` did run in the no-fail-fast run (7 ok). My separate `check-examples` run waited on b10x-build-slot (17G free, below its 20G floor) and I stopped it (PID 3046020).

**1. `git --no-pager diff --stat`**
```
 DESIGN.md                                       |   3 +-
 crates/modelspec-core/src/ir/print.rs           |  88 +++++-
 crates/modelspec-core/src/ir/training.rs        | 291 +++++++++++++++++-
 crates/modelspec-core/src/ir/verify.rs          | 319 +++++++++++++++++++-
 crates/modelspec-runtime/src/train/data.rs      | 301 +++++++++++++++++-
 crates/modelspec-runtime/src/train/init.rs      | 385 +++++++++++++++++++++++-
 crates/modelspec-runtime/src/train/mod.rs       |   3 +-
 crates/modelspec-runtime/src/train/optimizer.rs | 277 ++++++++++++++++-
 8 files changed, 1647 insertions(+), 20 deletions(-)
```
These are the implementor's changes; I changed no tracked file. Untracked: `crates/modelspec-core/tests/training_verify.rs` (the implementor's), plus my two files `crates/modelspec-core/tests/training_verify_adversary.rs` and `crates/modelspec-runtime/tests/train_adversary.rs`.

**2. Cases added (red output from each file run alone, before the suite)**

| File | Case | Now | Mutant that turns it red |
|---|---|---|---|
| training_verify_adversary.rs | a_param_read_whose_read_effect_names_another_parameter_is_refused | green | M1 (only this case) |
| training_verify_adversary.rs | an_update_no_node_defines_is_refused_by_its_own_rule | green | M2 (only this case) |
| training_verify_adversary.rs | a_gradient_tagged_optimizer_is_refused | RED | - |
| training_verify_adversary.rs | a_step_whose_learning_rate_is_not_positive_and_finite_is_refused | RED | - |
| training_verify_adversary.rs | verify_step_reports_the_output_layout_before_an_optimizer_value_used | RED | - |
| train_adversary.rs | a_zero_word_draws_a_finite_normal_from_u_2_to_the_minus_33 | green | M3 (implementor test also catches it) |
| train_adversary.rs | xavier_uniform_stays_inside_minus_a_to_a_after_rounding | RED | - |
| train_adversary.rs | a_zero_or_unresolved_dimension_is_refused_not_drawn | green | M4 (only this case) |

```
thread 'verify_step_reports_the_output_layout_before_an_optimizer_value_used' (2584431) panicked at crates/modelspec-core/tests/training_verify_adversary.rs:153:5:
Err(OptimizerValueUsed { node: ValueId(14), tag: Gradient, operand: ValueId(13) })
thread 'a_step_whose_learning_rate_is_not_positive_and_finite_is_refused' (2584429) panicked at crates/modelspec-core/tests/training_verify_adversary.rs:135:9:
optimizer lr NaN verifies: Ok(())
thread 'a_gradient_tagged_optimizer_is_refused' (2584427) panicked at crates/modelspec-core/tests/training_verify_adversary.rs:120:5:
the gradient %11 of @LinearReg.w is tagged optimizer and the step verifies: Ok(())
test result: FAILED. 2 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
thread 'xavier_uniform_stays_inside_minus_a_to_a_after_rounding' (2587627) panicked at crates/modelspec-runtime/tests/train_adversary.rs:49:5:
element 0 is 1.0954451561 = 1.09544515609741211; a is 1.09544511501033215
test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```
After the alone run, rustfmt joined one `assert!` in the lr case onto a single line. Behaviour is unchanged, but the layout case's panic line moved from 153 to 150.

Mutants, each applied to a copy outside the tree. Under every mutant all implementor suites stayed green, except M3, which `train::init::tests::standard_normal_is_box_muller_over_words_0_and_1` (init.rs:364) catches.
- M1: verify.rs:627 becomes "any `ParamRead`". A write ordered before the last read of `w` then verifies (`left: Ok(())`).
- M2: verify.rs:543-545 skips an undefined update. The step reports `OutputLayout` instead of `UndefinedUpdate`.
- M3: init.rs:100 drops the `+ 0.5`.
- M4: init.rs:109 drops `n > 0`. A zero dimension then panics at init.rs:122.

**3. Suite run (after the cases existed)**
`CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci`: build, `cargo fmt --check` and clippy `-D warnings` all passed. The test step stopped at the first red binary:
```
error: test failed, to rerun pass `-p modelspec-core --test training_verify_adversary`
task: Failed to run task "ci": task: Failed to run task "test": exit status 101
EXIT=201
```
`... b10x-build-slot cargo test --workspace --no-fail-fast` gave EXIT=101. Per binary: cli main 0, check 49, check_adversary 13, exit_codes 8, core lib 5, autodiff_adversary 6, no_vulkan 7, syntax 90, syntax_adversary 11, tensor_ir 26, tensor_ir_adversary 7, training_verify 30, **training_verify_adversary 2 passed / 3 failed**, runtime lib 28, cpu_adversary 6, cpu_golden 80, gradients 49, gradients_adversary 2, **train_adversary 2 passed / 1 failed**, doc-tests 0+0. The red output matches part 2 (layout case at :150). The "before" count of 417 is this run with my two binaries left out (8 cases).

**4. Judgement findings (on the working tree above)**

| # | file:line | Sev | Verdict | Origin | What reaches it |
|---|---|---|---|---|---|
| F1 | crates/modelspec-core/src/ir/verify.rs:627 | warning | CONFIRMED | introduced | Mutant M1. No implementor case pins the param half of `ReadEffectMismatch` (training_verify.rs:442 only covers a node that is not a `ParamRead`). |
| F2 | crates/modelspec-core/src/ir/verify.rs:545 | note | CONFIRMED | introduced | Mutant M2. `UndefinedUpdate` has no implementor case. |
| F3 | crates/modelspec-runtime/src/train/init.rs:142 | note | NEEDS-CHANGE | introduced | Seed 72369776 makes element 0 equal f32(a) > a, so the "U(-a, a)" doc is false. init.rs:261 and :280 assert `< a`. DESIGN § 2.3 (f64 rounded to f32) is met, so the doc is the likely fix. |
| F4 | crates/modelspec-core/src/ir/verify.rs:444 | note | INFEASIBLE | introduced | With `optimizer=true`, any node that depends on the seed may be tagged `Optimizer`, including a gradient output. Only a hand-built step reaches it; `attach_sgd` never does. |
| F5 | crates/modelspec-core/src/ir/verify.rs:502 | note | INFEASIBLE | introduced | `verify_step` never checks `Optimizer.lr` against its doc "positive and finite" (training.rs:131) or against the `Scalar(lr)` node that actually runs. `attach_sgd` refuses bad values (training.rs:194), so only a hand-built step reaches it. |
| F6 | crates/modelspec-core/src/ir/verify.rs:493 | note | NEEDS-CHANGE | introduced | The doc puts the output layout among the first checks; the code checks it last (:566), after `OptimizerValueUsed` and the update rules. |
| F7 | AGENTS.md:14 | note | NEEDS-CHANGE | introduced | It says DESIGN.md revision 0.4; the unit moves DESIGN.md to 0.5. |
| F8 | crates/modelspec-runtime/src/train/init.rs:109 | note | INFEASIBLE | introduced | Mutant M4. The zero-dimension guard is pinned by my case only. Source code cannot reach it, because the checker refuses empty shapes. |

**5. What I attacked and could not break**
- Philox4x32-10 matches the three Random123 known-answer vectors, the rounds and the key bump.
- The key, counter and stream-tag layout and the blake3 id match DESIGN § 2.3.
- The Box-Muller formula is right, and no zero reaches `ln` (M3 is killed).
- Draws past 2^32 elements are refused. Seed 42 and seed 43 give different draws, the streams are separate, and declaration order does not change any draw.
- `SecondWrite`, `WriteBeforeRead` and `WrongNextVersion` are each pinned by an implementor case.
- An unused parameter keeps its value and version, in both core and runtime.
- `sgd_step` refuses an illegal step before it runs, and refuses `VersionExhausted` without writing.
- E-DATA-001 and E-DATA-002 refuse as specified, and duplicate data names get E-NAME-005.
- Next to phase B, not checked: if a loss ignores its target, the synthetic loader would hit `UnexpectedInput` instead of returning a diagnostic. Whether that happens depends on unit 7's dead-code elimination.

**6. Paths written outside the worktree**
Everything is under `~/.cache/modelspec-wave-m0/m0-cpu-sgd-training/scratch/adversary/`:
- `probe/` (Rust seed-search probe, including its `target/`)
- `mutant/` (about 480M copy of the tree, including `target/`)
- `verify.rs.orig`, `init.rs.orig`, `mutate.sh`
- `mutant-M1.log` through `mutant-M4.log`
- `red-core-adversary.log`, `red-runtime-adversary.log`
- `suite-ci.log`, `suite-nff.log`

The worktree session lease I took (`adversary-m0-cpu-sgd-training-pass1`) is released.

**7.**
```findings
- file: crates/modelspec-core/src/ir/verify.rs
  line: 627
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: no implementor case pins the parameter of ReadEffectMismatch, so weakening it to any ParamRead lets a write ordered before the last read of w verify
- file: crates/modelspec-core/src/ir/verify.rs
  line: 545
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: UndefinedUpdate has no implementor case; skipping an undefined update leaves every implementor suite green
- file: crates/modelspec-runtime/src/train/init.rs
  line: 142
  category: property
  severity: note
  verdict: NEEDS-CHANGE
  origin: introduced
  message: xavier_uniform is documented and tested as U(-a, a) but rounding to f32 yields f32(a) > a at seed 72369776
- file: crates/modelspec-core/src/ir/verify.rs
  line: 444
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: verify_step accepts a parameter gradient tagged Optimizer because check_graph allows the tag on any seed-dependent node
- file: crates/modelspec-core/src/ir/verify.rs
  line: 502
  category: contract-drift
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: verify_step does not check that Optimizer.lr is positive and finite or equal to the Scalar(lr) node that runs
- file: crates/modelspec-core/src/ir/verify.rs
  line: 493
  category: contract-drift
  severity: note
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the documented order checks the output layout with the verify_training rules but the code checks it last, after OptimizerValueUsed
- file: AGENTS.md
  line: 14
  category: contract-drift
  severity: note
  verdict: NEEDS-CHANGE
  origin: introduced
  message: AGENTS.md names DESIGN.md revision 0.4 while the unit moves DESIGN.md to 0.5
- file: crates/modelspec-runtime/src/train/init.rs
  line: 109
  category: mutant
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: the zero-dimension guard is unpinned by the implementor suite and without it a zero dimension panics at init.rs:122
```
