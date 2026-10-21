---
format: aep.planning-md/3
id: review-result:adversary-observatory-input-adjoint-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:observatory-input-adjoint
relations:
- reviews: story:observatory-input-adjoint
revision: 1
---
I found no defect: 0 of my 10 new cases are red, and the training path matches the base commit on both models. The one finding is a gap in the implementor's tests, which my green case now covers.

```
unit: story:observatory-input-adjoint, commit 0d797e6 plus my untracked test files
verdict: note CONFIRMED (a test gap; no defect)
cases: executed 407→417, red 0
origin: introduced 1 / pre-existing 0 / undecided 0
wrote-outside-worktree: 9 paths, all in the assigned scratch directory (part 6)
needs-coordinator: whether to keep the 253-line copy of the base `autodiff.rs` my core cases read
```

**1. Diff stat.** `git --no-pager diff --stat` is empty: I changed no tracked file. `git status --short` shows only my three untracked test paths:
- `crates/modelspec-core/tests/adversary_input_adjoint.rs`
- `crates/modelspec-core/tests/adversary_input_adjoint_support/` (holds `base_autodiff.rs`, a verbatim `git show ab9ed27:` copy of the old `autodiff.rs`)
- `crates/modelspec-runtime/tests/adversary_input_adjoint.rs`

**2. Cases added (all green now)**

| Case | What it checks |
|---|---|
| core `adversary_training_program_equals_the_base_commit_for_every_model` | For linear regression and two-layer relu, new `differentiate` equals the base commit's, both as a value and as printed text. The `compile_training` nodes start with the base program. |
| core `..._for_hand_built_graphs` | The same equality on graphs with a parameter read 3 times, a rank-0 parameter on either side of `Mul`, `Add` and `Sub`, a trainable `mse` operand B and an unreached parameter. |
| core `adversary_training_refusals_equal_the_base_commit` | `NoGradientRule` and `LossNotScalar` read the same as at the base commit. |
| core `adversary_a_name_asked_twice_has_one_adjoint_value` | A repeated name gets the same value id and no second walk, for a read input and for an unread one (one zero). |
| core `adversary_a_target_that_is_the_input_itself` | The adjoint is the seed `Scalar(1.0)`. |
| core `adversary_an_input_read_three_times_sums_three_contributions` | Three uses of one input are summed with two `Add`s. |
| core `adversary_output_element_target_of_a_rank_0_prediction` | An empty index gives offset 0 and the result verifies. |
| runtime `adversary_shared_uses_and_scalar_broadcasts_match_finite_differences` | Adjoints of `x` and of a rank-0 `s` match finite differences at 1e-6 in f64 and 1e-3 in f32. `x` is used by `Mul x x`, by `MatMul` on both sides, by `mse` operand B, and through `Relu` and `Transpose`; `s` sits on both sides of `Mul`, `Add` and `Sub`. Also run with names `s, x, x`. |
| runtime `adversary_a_repeated_name_evaluates_at_every_position` | Two-layer relu with names `x, y, x`: every output position matches finite differences, in f64 and f32. |
| runtime `adversary_output_elements_past_the_first_match_finite_differences` | Rows `[3,0]`, `[7,0]` (linear) and `[9,0]`, `[15,0]` (two-layer): offset equals the row, the adjoint matches finite differences, and only that row of `x` is non-zero. |

The first run of the core file was red (`EXIT=101`) at the third assertion of the every-model case. That was my mistake: I compared against `compiled.training.training`, which includes the SGD update nodes. I replaced it with the prefix check. It is not a finding.

**Mutant probe.** I made a copy of the current `autodiff.rs` in scratch with `resolved.insert(value, g);` (line 292) deleted, and ran my repeated-unread-name assertion against it:
```
assertion `left == right` failed: unread input
  left: ValueId(6)
 right: ValueId(7)
test result: FAILED. 0 passed; 1 failed ... EXIT=101
```
I then deleted the probe file. I did not run the implementor's own tests against the mutant. Reading them, none asks for an unread name twice, and `assert_layout` on `["x","x"]` checks only names and types.

**3. Suite run, after the cases existed**

| Command | Summary | Exit |
|---|---|---|
| `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot cargo test -p modelspec-core -p modelspec-runtime` | every binary `ok`; 417 passed, 0 failed | 0 |
| `b10x-build-slot cargo test -p modelspec-cli --test train_linear` | `ok. 13 passed; 0 failed` | 0 |
| `b10x-build-slot cargo clippy -p modelspec-core -p modelspec-runtime --all-targets -- -D warnings` | clean | 0 |
| `cargo fmt -p modelspec-core -p modelspec-runtime -- --check` | clean after I formatted my two files (13 diffs, all in them) | 0 |

The `before` count of 407 is that same run minus my 7 + 3 cases. It matches the implementor's `gate-test.log`.

**4. Findings**

| file:line | Verdict | Origin | What was measured | What reaches it |
|---|---|---|---|---|
| `crates/modelspec-core/src/autodiff.rs:292` | CONFIRMED, note | introduced | Deleting the zero-reuse line breaks the "a name asked twice has one adjoint" claim. My case catches it; by reading, the implementor's 24 cases do not. | Any caller that repeats an unread name. No such caller exists yet. |

**5. Attacked and not broken**
- **Shared-use accumulation:** a triple read, both `MatMul` operands, `mse` operand B, and rank-0 inputs either side of `Mul`/`Add`/`Sub` all match finite differences in f64 and f32.
- **Training path unchanged:** equal to the base commit for both models and two hand-built graphs, including the refusal messages.
- **Out-of-range index and rank:** the implementor's refusal tests are exact. A u64 overflow of the offset cannot happen, because the verifier refuses shapes whose element count overflows.
- **Rank > 0 targets:** refused (implementor test). Elements past `[0, 0]` match finite differences.
- **Non-differentiable points:** an input that reaches the target only through `ReluGrad` or `BroadcastScalar` is refused with `NoGradientRule` rather than given a zero adjoint. This needs a hand-built forward program: `lower.rs` never emits `BroadcastScalar`.
- **Repeated names:** same value id, and evaluated correctly at every output position.
- **`mse` operand B:** gets an adjoint only when it depends on a named input.

**6. Paths written outside the worktree**, all under `~/.cache/modelspec-wave-observatory-1/observatory-input-adjoint/scratch/`: `adv-core-alone.log`, `adv-runtime-alone.log`, `mutant_autodiff.rs`, `mutant-probe.log`, `adv-suite.log`, `adv-cli.log`, `adv-clippy.log`, `adv-fmt.log`, `adv-after-fmt.log`.

**7. Findings block**
```findings
- file: crates/modelspec-core/src/autodiff.rs
  line: 292
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: deleting `resolved.insert(value, g)` (an unread name asked twice gets two zero adjoints) survives the unit's 24 cases by inspection and is caught by adversary_a_name_asked_twice_has_one_adjoint_value
```