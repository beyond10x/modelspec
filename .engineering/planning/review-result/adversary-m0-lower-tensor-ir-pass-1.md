---
format: aep.planning-md/3
id: review-result:adversary-m0-lower-tensor-ir-pass-1
kind: review-result
status: active
title: Adversary pass 1 on m0-lower-tensor-ir
relations:
- reviews: story:m0-lower-tensor-ir
revision: 1
---
unit: story:m0-lower-tensor-ir, pass 1. Covers the working tree on branch impl/m0-lower-tensor-ir (base aa020fa), whose unit is uncommitted, plus the new file crates/modelspec-cli/tests/ir_tensor_adversary.rs
verdict: CONFIRMED
cases: executed 391→397, red 2
origin: introduced 10 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory tree (part 6)
needs-coordinator: should E-LOWER-001 move into the checker, so `check` refuses a model without `forward` too, or should lowering stop requiring `forward`? The checker files and DESIGN.md are outside this unit's file assignment.

**1. `git --no-pager diff --stat`**
```
 AGENTS.md                                |   5 +-
 crates/modelspec-cli/src/ir.rs           |  48 +++++-
 crates/modelspec-cli/tests/exit_codes.rs |   4 +-
 crates/modelspec-core/src/compile.rs     | 151 ++++++++++++++++-
 crates/modelspec-core/src/lower.rs       | 273 ++++++++++++++++++++++++++++++-
 crates/modelspec-core/src/passes/dce.rs  | 102 +++++++++++-
 6 files changed, 569 insertions(+), 14 deletions(-)
```
Every tracked change above is the implementor's uncommitted work. I changed no tracked file. lower.rs, dce.rs, compile.rs and ir.rs are byte-identical (`cmp`) to the copies I took when I started. The AGENTS.md change equals the implementor's `scratch/agents-md-state.patch`, written at 15:26, before I started at 15:32. `git status --short` shows one untracked path of mine: `crates/modelspec-cli/tests/ir_tensor_adversary.rs`. The other untracked paths, `tests/golden/` and `tests/ir_tensor.rs`, are the implementor's. No non-test path is mine.

**2. Cases added** (crates/modelspec-cli/tests/ir_tensor_adversary.rs, 6 tests)

| Test | Asserts | Now |
|---|---|---|
| `ir_with_an_unwritable_stdout_exits_1_without_a_panic` (linux) | `ir --stage tensor` with stdout on /dev/full exits 1 and prints no panic | RED |
| `check_and_ir_agree_on_a_model_without_a_forward_function` | `forward` is renamed to `predict`; `check` and `ir` give the same exit code | RED |
| `the_lowered_program_computes_the_source_on_the_cpu_interpreter` | A source using Sub, MatMul, Relu, Transpose, Mean, scalar literals and a dead let: loss 51.875, prediction [[-9],[0]] | green |
| `a_target_declared_before_its_input_is_the_first_input_node` | Exact print: `Input "y"` is %0 and `Input "x"` is %1 | green |
| `a_scalar_node_carries_the_span_of_its_literal` | A Scalar node has value 0.5 and the HIR literal's span | green |
| `every_edit_check_accepts_lowers_verifies_runs_and_differentiates` | Over 10,000 token edits; for each one `check` accepts: lowers or gives only E-LOWER-001, verifies, keeps the prediction live, DCE matches an independent reachability, deterministic, runs on the interpreter, `differentiate` + `verify_training` succeed | green |

The test writes its fixture to `target/tmp/ir_tensor_adversary/predict.mspec`, inside the tree, through CARGO_TARGET_TMPDIR.

Red output from the solo run, captured before the suite. Command: `cargo test -p modelspec-cli --test ir_tensor_adversary`, exit 101, 4 passed, 2 failed. I replaced the tree's absolute prefix with nothing.
```
thread 'check_and_ir_agree_on_a_model_without_a_forward_function' (2201501) panicked at crates/modelspec-cli/tests/ir_tensor_adversary.rs:102:5:
assertion `left == right` failed: check exits Some(0) with stderr ""; ir exits Some(1) with stderr "E-LOWER-001: the loss of train block `Fit` does not call `model.forward`\n  target/tmp/ir_tensor_adversary/predict.mspec:15:12\n  the prediction is the value of `model.forward(...)` in the loss\n"
  left: Some(0)
 right: Some(1)

thread 'ir_with_an_unwritable_stdout_exits_1_without_a_panic' (2201503) panicked at crates/modelspec-cli/tests/ir_tensor_adversary.rs:86:5:

thread 'main' (2201505) panicked at /rustc/b940084d7eb6a299eb4bfeb8e34901bc051e7ac4/library/std/src/io/stdio.rs:1168:9:
failed printing to stdout: No space left on device (os error 28)
```
The first `task ci` run failed at lint on my own file: `clippy::collapsible_if` at ir_tensor_adversary.rs:284. I fixed it in that file only. The change is lint-only, and both cases fail the same way afterwards.

**3. Suite run** (after the cases existed)

Command: `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci`. Passing `test … ok` lines and blank lines are removed; the full log is at `scratch/adversary/suite.log`.
```
task: [build] cargo build --workspace
task: [lint] cargo fmt --check
task: [lint] cargo clippy --workspace --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.04s
task: [test] cargo test --workspace
     Running tests/check.rs            test result: ok. 49 passed
     Running tests/check_adversary.rs  test result: ok. 13 passed
     Running tests/exit_codes.rs       test result: ok. 8 passed
     Running tests/ir_tensor.rs        test result: ok. 17 passed
     Running tests/ir_tensor_adversary.rs
test ir_with_an_unwritable_stdout_exits_1_without_a_panic ... FAILED
test check_and_ir_agree_on_a_model_without_a_forward_function ... FAILED
[same two panics as part 2]
test result: FAILED. 4 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.10s
error: test failed, to rerun pass `-p modelspec-cli --test ir_tensor_adversary`
task: Failed to run task "ci": task: Failed to run task "test": exit status 101
EXIT=201
```
- Lint was clean on this run.
- `cargo test` stopped at the first red test binary after 93 cases.
- To count every case, `scratch/adversary/rest.sh` ran the rest through b10x-build-slot:
  - `cargo test --workspace --no-fail-fast`: exit 101, only `-p modelspec-cli --test ir_tensor_adversary` failed;
  - `core-no-vulkan`: exit 0, 7 passed;
  - `check-examples`: exit 0.
- That makes 397 cases executed with 2 red.
- The 391 before-count comes from the implementor's green gate log `scratch/gate.log` (384 + 7), which is 397 minus my 6.

**4. Judgement findings**

| file:line | Verdict / origin | What was measured | What reaches it |
|---|---|---|---|
| crates/modelspec-cli/src/ir.rs:43 | CONFIRMED / introduced | `print!` panics on a failed stdout write. Exit 101 with /dev/full, and 5 of 5 runs exit 101 piped into `true` ("Broken pipe (os error 32)"). AGENTS.md:34 says exit codes are 0/1/2. | `modelspec ir … \| <reader that exits early>`, a full disk, a closed stdout |
| crates/modelspec-core/src/lower.rs:125 | CONFIRMED / introduced | E-LOWER-001 makes `forward` a required hook: `check` exits 0 and `ir` exits 1 on the same file. DESIGN.md:49 says "`forward` is a regular pure function, not a magical runtime hook." At base, `ir` failed every file as "not implemented", so this disagreement is new. | Any model that names its function something other than `forward` |
| crates/modelspec-core/src/lower.rs:202 | CONFIRMED / introduced | Scalar value (1.0, negated) and span (None) mutants survive the implementor's suite. My green cases now kill all three. | Every literal in a loss or forward |
| crates/modelspec-core/src/lower.rs:210 | CONFIRMED / introduced | Swapping Sub operands, or MatMul operands when both have the same type, survives the implementor's suite. My interpreter case now kills both. | Every `sub(a, b)` and square `matmul` |
| crates/modelspec-core/src/lower.rs:92 | CONFIRMED / introduced | Ordering inputs by role (inputs, then targets) instead of source order survives the implementor's suite. My target-first case now kills it. | A target declared before an input |
| crates/modelspec-core/src/lower.rs:61 | CONFIRMED / introduced | Negating `is_compiler_bug` survives every case, mine included. It is a public function with no caller. | nothing found |
| crates/modelspec-core/src/lower.rs:127 | CONFIRMED / introduced | Moving the E-LOWER-001 span from the loss to the train block survives every case. Nothing asserts the diagnostic's location. | Every E-LOWER-001 report |
| crates/modelspec-core/src/lower.rs:219 | CONFIRMED / introduced | For `model.forward(model.forward(x))` the prediction is %3 (the inner call) while `Mse` reads %4 (scratch probe). lower.rs:37 documents "first … in evaluation order". | A model whose forward maps a shape to itself, called twice in the loss |
| crates/modelspec-core/src/passes/dce.rs:24 | INFEASIBLE / introduced | DCE drops an unread Input from `inputs` but keeps unread params in the table. A host that passes every declared input gets `UnexpectedInput { name: "z" }` (scratch probe). | Nothing today: `train` is unimplemented; it is the first caller that will bind data by declared name |
| AGENTS.md:17 | CONFIRMED / introduced | Edited although the brief lists AGENTS.md as "not yours". The new text is accurate, and a patch of the edit also sits in the implementor's scratch. | Scope of this unit |

**5. Attacked and could not break**
- DCE renumbering, liveness and keeping the prediction: matches an independent worklist reachability over more than 10,000 accepted edits.
- One shared ParamRead per param; inputs first.
- Determinism: a second compile is byte-identical.
- `differentiate` + `verify_training` on every lowered program that `check` accepts.
- The verifier and prediction-check paths in compile.rs: mutants killed by the implementor's unit tests.
- The implementor's suite already killed the arg-reverse, let-last, op-span, dce-identity and dce-keep-inputs mutants.
- `--stage` missing or with an unknown value exits 2.
- Missing file exits 1.

**6. Paths written outside the worktree**, all under `~/.cache/modelspec-wave-m0/m0-lower-tensor-ir/scratch/adversary/`:
- `copy/`: a copy of the tree without target, .git or .engineering. It includes `copy/crates/modelspec-cli/tests/ir_tensor_adversary.rs` (the pre-lint version) and `copy/crates/modelspec-cli/tests/scratch_probe_adversary.rs`.
- `copy/target/`: a 214M build directory.
- `pristine/lower.rs`, `pristine/dce.rs`, `pristine/compile.rs`, `pristine/ir.rs`.
- Scripts: `probe.sh`, `run-probes.sh`, `survivors.sh`, `rest.sh`.
- Logs: `run-probes.out`, `survivors.out`, `probe.log`, 14 files named `probe-<mutant>.log`, `adversary-alone.log`, `scratch-probe.log`, `suite.log`, `rest.log`.

I released my worktree lease (`worktree hook session-end`, exit 0).

**7.**
```findings
- file: crates/modelspec-cli/src/ir.rs
  line: 43
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: print! panics with exit 101 when stdout is closed or full (a closed pipe reproduced 5 of 5), breaking the documented 0/1/2 exit-code contract
- file: crates/modelspec-core/src/lower.rs
  line: 125
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: E-LOWER-001 makes forward a required hook, so check exits 0 and ir exits 1 on the same file, against DESIGN.md line 49
- file: crates/modelspec-core/src/lower.rs
  line: 202
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: scalar value and span mutants survived the implementor's suite; ir_tensor_adversary now kills them
- file: crates/modelspec-core/src/lower.rs
  line: 210
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: swapped Sub and square-MatMul operands survived the implementor's suite; the interpreter case now kills them
- file: crates/modelspec-core/src/lower.rs
  line: 92
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: ordering inputs by role instead of source order survived the implementor's suite; the target-first case now kills it
- file: crates/modelspec-core/src/lower.rs
  line: 61
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: negating is_compiler_bug survives every case because the public fn has no caller and no test
- file: crates/modelspec-core/src/lower.rs
  line: 127
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: moving the E-LOWER-001 span to the train block survives every case because nothing asserts the diagnostic location
- file: crates/modelspec-core/src/lower.rs
  line: 219
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: for model.forward(model.forward(x)) the prediction is the inner call %3 while the loss reads the outer %4
- file: crates/modelspec-core/src/passes/dce.rs
  line: 24
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: DCE drops unread inputs but keeps unread params, so a host passing every declared input gets UnexpectedInput, and no caller binds inputs yet
- file: AGENTS.md
  line: 17
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: AGENTS.md was edited although the brief lists it as not this unit's file
```
