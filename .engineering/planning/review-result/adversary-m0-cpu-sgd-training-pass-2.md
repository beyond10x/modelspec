---
format: aep.planning-md/3
id: review-result:adversary-m0-cpu-sgd-training-pass-2
kind: review-result
status: active
title: Adversary pass 2 (phase B) on story:m0-cpu-sgd-training
relations:
- reviews: story:m0-cpu-sgd-training
revision: 1
---
unit: story:m0-cpu-sgd-training, working tree = 0228365 plus the uncommitted phase-B change, plus one adversary test file
verdict: CONFIRMED (no blocker)
cases: executed 495→509, red 1
origin: introduced 5 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory (~/.cache/modelspec-wave-m0/m0-cpu-sgd-training/scratch/adversary-b/)
needs-coordinator: none

1. `git --no-pager diff --stat`. These are the implementor's uncommitted files; I changed none of them. My only change is the untracked file `crates/modelspec-cli/tests/train_linear_adversary.rs`.
```
 crates/modelspec-cli/src/ir.rs               |  67 ++++++----
 crates/modelspec-cli/src/main.rs             |   3 -
 crates/modelspec-cli/src/train.rs            | 137 ++++++++++++++++++-
 crates/modelspec-cli/tests/exit_codes.rs     |   4 +-
 crates/modelspec-cli/tests/ir_tensor.rs      |  11 +-
 crates/modelspec-core/src/compile.rs         | 131 +++++++++++++++++--
 crates/modelspec-runtime/src/train/data.rs   | 188 +++++++++++++++++++++-----
 crates/modelspec-runtime/src/train/mod.rs    |   3 +-
 crates/modelspec-runtime/src/train/runner.rs | 189 ++++++++++++++++++++++++++-
 9 files changed, 648 insertions(+), 85 deletions(-)
?? crates/modelspec-cli/tests/train_linear_adversary.rs
```

2. Cases added in `crates/modelspec-cli/tests/train_linear_adversary.rs` (14 cases, 1 red).

| case | asserts | now | kills mutant |
|---|---|---|---|
| `train_with_stdout_closed_exits_1` | with file descriptor 1 closed, exit 1 and "cannot write the run" (train.rs:16 says so) | **red** | - |
| `one_step_at_lr_0_05_and_seed_7_is_one_update_at_0_05` | `lr`, `seed` and `steps` are read from the source; `param` equals one f64 SGD update at 0.05 | green | M1, M3 |
| `the_compiled_step_matches_the_forward_and_central_differences` | 2-parameter model: each gradient element matches the central difference, loss output equals the forward loss, update equals `p - 0.25 g` | green | M1 |
| `the_loss_lines_follow_the_source_steps` | steps 120, 50 and 49 print lines {0,50,100,120}, {0,50} and {0,49} | green | M3 |
| `vulkan_is_refused_before_the_file_is_read` | a missing file or a broken file with vulkan still gives exactly the E-TARGET-001 stderr | green | M2 |
| `train_with_a_closed_stdout_pipe_exits_1_without_a_panic` | reader end of the stdout pipe dropped: exit 1, "cannot write the run" | green | M4 |
| `train_with_a_full_stdout_exits_1_without_a_panic` | stdout is /dev/full: exit 1 | green | M4 |
| `every_printed_loss_and_the_final_w_follow_f64_sgd_from_the_printed_init` | all 11 loss lines within 1e-3 relative of an f64 SGD run; final w within 1e-5 | green | M7 |
| `a_target_the_loss_does_not_read_is_refused_at_its_declaration` | real source: exact E-DATA-003 stderr at `:12:5` | green | M5 |
| `the_device_line_is_ignored` | `device = vulkan(0);` with `--backend cpu` prints the same bytes as the example | green | - |
| `a_target_declared_before_the_input_trains_the_same` | target declared before input prints the same bytes | green | - |
| `a_loss_that_reads_no_parameter_trains_and_keeps_the_parameter` | exit 0, init equals param, `ir --stage training` exits 0 | green | M7 |
| `a_parameter_read_twice_trains_to_the_teacher` | `add(matmul(x,w), matmul(x,w))` reaches the teacher within 1e-2 | green | - |
| `every_checked_edit_trains_one_step_or_is_refused_with_a_user_code` | every one-token edit of 3 sources: no E-INTERNAL, no panic, no run error | green | - |

The red case, run alone and before the suite (`cargo test -p modelspec-cli --test train_linear_adversary -- --exact train_with_stdout_closed_exits_1`):
```
thread 'train_with_stdout_closed_exits_1' (4056849) panicked at crates/modelspec-cli/tests/train_linear_adversary.rs:379:5:
assertion `left == right` failed: stderr: ""
  left: Some(0)
 right: Some(1)
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 13 filtered out; finished in 0.04s
```
The first run of the whole file was 13 passed and 1 failed (this case). That run swallowed the panic message because my fuzz case swapped the global panic hook. I removed the swap and then ran rustfmt on my file only.

Mutation probes ran on a copy (`scratch/adversary-b/mutant`, `mutate.sh`) with `cargo test --workspace --no-fail-fast`:

| mutant | change | implementor's suite | my file |
|---|---|---|---|
| M1 | compile.rs:49 `config.lr` replaced by `0.1` | **survives** | killed |
| M2 | train.rs:57 vulkan refused only when the file compiles | **survives** | killed |
| M3 | train.rs:129 `\|\| k == last` dropped | **survives** | killed |
| M4 | train.rs:99 write error mapped to exit 0 | **survives** | killed |
| M5 | data.rs:104 E-DATA-003 checks inputs only | killed (data unit test) | killed |
| M6 | runner.rs:66 final loss copied from the previous step | killed (2 runner tests) | - |
| M7 | runner.rs:58 initial state = teacher | killed (train_linear and 3 runner tests) | killed |

3. Suite run, after the cases existed.
- `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci`: exit 201. build, fmt and clippy passed. The test step stopped at `train_linear_adversary`: `test train_with_stdout_closed_exits_1 ... FAILED`, `test result: FAILED. 13 passed; 1 failed`.
- The same with `cargo test --workspace --no-fail-fast`: exit 101, 508 passed and 1 failed across all binaries; every other binary is ok.
- The "before" count of 495 is this same run with my file's 14 cases taken out.

4. Findings (tree: 0228365 plus the uncommitted change).

| file:line | severity | verdict | origin | what was measured | what reaches it |
|---|---|---|---|---|---|
| crates/modelspec-cli/src/train.rs:94 | note | CONFIRMED | introduced | with fd 1 closed, `train` exits 0 and writes nothing to stderr. Rust's std treats EBADF on stdout as a successful write. This contradicts train.rs:16 "output that cannot be written ... exits 1". `ir` (ir.rs:57) does the same at the base, but ir.rs's doc only promises "a closed pipe, a full device". | a shell `>&-`; nothing in the repo does it. Fix: narrow the train.rs:16 doc the way ir.rs words it. |
| crates/modelspec-core/src/compile.rs:49 | warning | CONFIRMED | introduced | M1 (lr hardcoded to 0.1) survives the implementor's suite: no existing case uses an lr other than 0.1 | every source with a non-0.1 lr |
| crates/modelspec-cli/src/train.rs:57 | warning | CONFIRMED | introduced | M2 survives: the vulkan test uses a file that compiles, so "before the file is read" is not pinned | `--backend vulkan` on a missing or broken file |
| crates/modelspec-cli/src/train.rs:129 | warning | CONFIRMED | introduced | M3 survives: 500 is a multiple of 50, so the "and the last step" line is never tested | any `steps` that is not a multiple of 50 |
| crates/modelspec-cli/src/train.rs:99 | warning | CONFIRMED | introduced | M4 survives: `train` has no case for a stdout it cannot write | `train ... \| head -c0`, a full disk |

The fix for the four warnings is to adopt the matching green cases into train_linear.rs.

5. Attacked and not broken:
- **Gradient sign, lr applied once, mse scale 2/8, step counter:** matches the f64 SGD run over 500 steps.
- **Gradients and updates on a 2-parameter model:** match central differences and `p - lr g`.
- **Teacher, init and data streams are separate:** M7 is killed and the init line equals the Init-stream draw.
- **Determinism:** two runs print identical bytes (implementor's case). Seed, steps and lr are all read from the source.
- **E-DATA-001/002/003:** order and spans hold, including an unread target in a real source.
- **The `device` line:** ignored, as specified.
- **Verifier:** `sgd_step` calls `verify_step` on every step, and the runner does not bypass it (by reading).
- **Training golden:** consistent with `print_step` and the § 3.1 tag definitions, and compared exactly in 2 cases.
- **One-token edit fuzz:** no compiler bug reached from a source that `check` accepts. The test asserts more than 100 accepted and more than 100 compiled; exact counts are not printed.

6. Paths written outside the worktree:
- `~/.cache/modelspec-wave-m0/m0-cpu-sgd-training/scratch/adversary-b/` contains:
  - `mutant/`: a source copy plus its own `target/`, 430M;
  - `mutate.sh`;
  - `compile.rs.orig`, `train.rs.orig`, `runner.rs.orig`, `data.rs.orig`;
  - `train_linear_adversary.rs`: the pre-rustfmt copy;
  - logs: `mutant-M1.log` to `mutant-M7.log`, `red-cli.log`, `red-cli-closed.log`, `suite-ci.log`, `suite-nff.log`.
- Inside the worktree's own target: `target/tmp/train_linear_adversary/` (the test's `CARGO_TARGET_TMPDIR` sources).
- Session lease `adversary-b-m0-cpu-sgd-training`: acquired and released.

```findings
- file: crates/modelspec-cli/src/train.rs
  line: 94
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: with file descriptor 1 closed train exits 0 with empty stderr because std swallows EBADF on stdout, contradicting the module doc at train.rs:16 that unwritable output exits 1
- file: crates/modelspec-core/src/compile.rs
  line: 49
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: replacing config.lr with 0.1 survives the implementor's suite because no case trains or compiles a source with another learning rate
- file: crates/modelspec-cli/src/train.rs
  line: 57
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: moving the vulkan refusal after reading and compiling the file survives the implementor's suite because its vulkan case uses a file that compiles
- file: crates/modelspec-cli/src/train.rs
  line: 129
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: dropping the last-step loss line survives the implementor's suite because 500 is a multiple of 50
- file: crates/modelspec-cli/src/train.rs
  line: 99
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: mapping a stdout write error to exit 0 survives the implementor's suite because train has no closed-pipe or full-device case
```
