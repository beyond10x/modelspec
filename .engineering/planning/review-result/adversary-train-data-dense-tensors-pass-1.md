---
format: aep.planning-md/3
id: review-result:adversary-train-data-dense-tensors-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:train-data-dense-tensors
relations:
- reviews: story:train-data-dense-tensors
revision: 1
---
unit: story:train-data-dense-tensors, working tree of impl/train-data-dense-tensors on d7d5fca (uncommitted, sources at 06:33:02)
verdict: NEEDS-CHANGE
cases: executed 646→659, red 2
origin: introduced 4 / pre-existing 0 / undecided 0
wrote-outside-worktree: 9 paths (part 6)
needs-coordinator: whether `--init` should refuse a missing directory or a directory that matches no parameter is a contract call (spec/domains/data.yaml ParamInitDirectory); the two red cases assert "refuse with E-DATA-007"

## 1. Diff stat

`git --no-pager diff --stat d7d5fca` is the implementor's 88 files, unchanged by me. My only path in the tree is one new untracked test file:

```
?? crates/modelspec-cli/tests/adversary_dense_1.rs
```

No implementation file, spec, doc or store file touched.

## 2. Cases added: crates/modelspec-cli/tests/adversary_dense_1.rs (13)

| Case | Asserts | Now |
|---|---|---|
| an_init_directory_that_does_not_exist_is_refused (:124) | `--init <nonexistent>` exits 1 with E-DATA-007 and empty stdout | RED |
| an_init_directory_whose_files_name_no_parameter_is_refused (:133) | `--init` dir holding only `Lin.w.f32` (full path, not stem) exits 1 with E-DATA-007 | RED |
| one_trailing_byte_on_a_data_file_is_e_data_007_with_empty_stdout | 33-byte y.f32 refused, stdout empty | green |
| a_zero_length_input_file_is_e_data_007_with_empty_stdout | 0-byte x.f32 refused | green |
| an_empty_order_file_is_e_data_008_with_empty_stdout | 0-byte order.u32 is E-DATA-008 | green |
| an_order_of_steps_times_b_minus_one_is_e_data_008_with_empty_stdout | 5 indices for steps·B = 6 refused | green |
| an_order_index_equal_to_n_past_the_run_is_e_data_008 | index N at position 7 (> steps·B) refused | green |
| an_order_of_index_n_minus_one_everywhere_is_accepted | index N-1 accepted, step 0 loss 0.5 | green |
| an_init_file_one_value_too_long_is_e_data_007_with_empty_stdout | 5 values for a 4-value param refused | green |
| a_stray_file_beside_the_init_files_is_ignored | notes.txt and order.u32 in init dir do not disturb `w` | green |
| the_eval_directory_order_file_is_not_read | malformed eval/order.u32 ignored | green |
| eval_without_data_runs_and_mismatched_eval_counts_are_e_data_007 | `--eval` alone prints a last eval line; eval count mismatch refused | green |
| eval_ties_take_the_first_maximum_and_divide_by_the_sample_count | tied rows take index 0, accuracy 0.75 = 3/4 samples | green |

Red output, the file alone, first run (`b10x-build-slot cargo test -p modelspec-cli --test adversary_dense_1`; full log `scratch/adv1-red.log`):

```
---- an_init_directory_whose_files_name_no_parameter_is_refused stdout ----
assertion `left == right` failed: exit Some(0)
stdout:
init @Lin.w: Tensor<f32, [2, 2]> = [-0.4053068, 0.6607273, -0.57972974, -0.8513097]
step 0 loss 1.4079305
step 3 loss 3.0049808
param @Lin.w: Tensor<f32, [2, 2]> = [-0.06993173, 0.62814516, -0.38600633, -0.617207]
stderr:
  left: Some(0)
 right: Some(1)

---- an_init_directory_that_does_not_exist_is_refused stdout ----
assertion `left == right` failed: exit Some(0)
(same stdout: the drawn xavier init, exit 0, empty stderr)

test result: FAILED. 11 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

## 3. Suite run (after part 2)

`CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot cargo test --workspace --no-fail-fast` (log `scratch/adv1-suite.log`):

```
EXIT=101
passed=657 failed=2
test an_init_directory_whose_files_name_no_parameter_is_refused ... FAILED
test an_init_directory_that_does_not_exist_is_refused ... FAILED
error: 1 target failed:
    `-p modelspec-cli --test adversary_dense_1`
```

646 before is the implementor's count; 659 = 646 + my 13. All 646 of theirs pass.

Acceptance 4 rerun: release rebuilt via `b10x-build-slot cargo build --release -p modelspec-cli` (exit 0, 4.95 s, df 36G free), then `run-mnist.sh`:

```
exit: 0
wall: 1.053703543 s
load before: 23.24 17.59 16.62
step 0 loss 0.33230302
step 1800 loss 0.026719691
eval loss=0.026219031689688563 accuracy=0.9088
```

stdout is byte-identical to the implementor's run (`cmp`, 42 lines). Reproduces.

## 4. Findings

| # | file:line | What was measured | What reaches it | Verdict | Origin |
|---|---|---|---|---|---|
| 1 | crates/modelspec-runtime/src/train/init.rs:221 | `--init <nonexistent dir>` exits 0 and trains from the xavier draw, stderr empty (red case :124) | a mistyped `--init` path in the documented benchmark command (`run-mnist.sh`); the result looks plausible | NEEDS-CHANGE | introduced |
| 2 | crates/modelspec-runtime/src/train/init.rs:211 | an `--init` dir whose files match no parameter (`Lin.w.f32` instead of `w.f32`) exits 0 and silently uses the initializer (red case :133) | naming the file by the canonical path, which is what the `init @Lin.w` line shows the user | NEEDS-CHANGE | introduced |
| 3 | crates/modelspec-runtime/src/train/runner.rs:168 | eval rows `[NaN, 1]` and `[Inf, 0]` count as correct: `eval loss=NaN accuracy=1` exit 0; NaN in `--data` trains to `param = [NaN, …]` exit 0 (probe `scratch/adv1-nan.sh`) | a broken preparation step writing non-finite values; spec and DESIGN say nothing about non-finite data | CONFIRMED | introduced |
| 4 | crates/modelspec-runtime/src/train/runner.rs:8 | module doc still says "Nothing in a run depends on anything but the compiled program and its seed"; `run_with` now depends on `--data`/`--init` files | the next reader of runner.rs | CONFIRMED | introduced |

Fix named, not applied: for 1 and 2, refuse with E-DATA-007 when the `--init` directory cannot be read, and when it holds no `<stem>.f32` for any parameter (or list stems found vs expected). Both need a line in ParamInitDirectory and DESIGN § 8.1. If the coordinator keeps the current contract ("a param with no file keeps its initializer"), the two red cases go and 1 and 2 become notes.

## 5. Attacked, not broken

- Endianness: LE throughout; fixtures pin it through exact losses.
- Length not a multiple of 4, one trailing byte, zero-length file: E-DATA-007, stdout empty.
- order.u32: empty, steps·B−1, index N inside and past the run: E-DATA-008; N−1 accepted; partial index: E-DATA-007.
- Step count vs order: exactly steps·B accepted, final loss wraps as DESIGN § 8 says.
- `--init` wrong size, short and long: E-DATA-007; stray files ignored.
- Eval: first-max ties, denominator N, B not dividing N (E-DATA-009), eval order.u32 not read, `--eval` without `--data`.
- Every refusal before step 0 with empty stdout (007, 008, 009, including init/eval read after data).
- Spec (data.yaml), DESIGN § 8/§ 8.1, CLI help, diagnostics page and CHANGELOG agree with the code on every code and layout rule I checked.
- MNIST result reproduces bit-identically from the current tree.

## 6. Paths written outside the worktree

- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/adversary-1.md
- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/adv1-red.log
- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/adv1-suite.log
- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/adv1-release-build.log
- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/adv1-nan.sh
- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/adv1-nan/ (fixture dir)
- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/adv1-implementor-mnist-run.stdout and .stderr (copies of the implementor's run, saved before mine)
- ~/.cache/modelspec-wave-m1-a/train-data-dense-tensors/scratch/mnist-run.stdout and .stderr (overwritten by `run-mnist.sh`; identical content)

Inside the worktree, besides the test file: `target/tmp/adversary_dense_1/` (test fixtures) and the rebuilt `target/release/modelspec`.

## 7. Findings block

```findings
[
 {
  "file": "crates/modelspec-runtime/src/train/init.rs",
  "line": 221,
  "category": "boundary",
  "severity": "warning",
  "verdict": "NEEDS-CHANGE",
  "origin": "introduced",
  "message": "--init naming a directory that does not exist exits 0 and trains from the drawn initializer with no diagnostic"
 },
 {
  "file": "crates/modelspec-runtime/src/train/init.rs",
  "line": 211,
  "category": "boundary",
  "severity": "warning",
  "verdict": "NEEDS-CHANGE",
  "origin": "introduced",
  "message": "an --init directory whose files match no parameter stem (for example Lin.w.f32) is silently ignored and the run exits 0"
 },
 {
  "file": "crates/modelspec-runtime/src/train/runner.rs",
  "line": 168,
  "category": "judgement",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "non-finite data is not refused and an eval row with a NaN prediction counts as correct, giving eval loss=NaN accuracy=1"
 },
 {
  "file": "crates/modelspec-runtime/src/train/runner.rs",
  "line": 8,
  "category": "contract-drift",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "the module doc still says a run depends only on the compiled program and its seed, which --data and --init make false"
 }
]
```
