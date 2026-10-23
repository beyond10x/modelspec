---
format: aep.planning-md/3
id: review-result:m1-g-nonfinite-adversary-1
kind: review-result
status: active
title: 'm1-g unit 2 adversary pass 1: 1 red (doc contract), 1 note'
relations:
- reviews: story:train-refuses-non-finite-loss
revision: 1
---
```
unit: story:train-refuses-non-finite-loss, tree modelspec-impl-m1g-nonfinite at cc66345 (base 2449897) plus 1 untracked test file
verdict: NEEDS-CHANGE
cases: executed 913→918, red 1
origin: introduced 1 / pre-existing 0 / undecided 0
needs-coordinator: story text says `error[E-TRAIN-001]`, which no diagnostic in the repo uses; a follow-up for the same check on `--backend vulkan`
```

1. **Cases added**, `crates/modelspec-cli/tests/adversary_train_non_finite.rs`:

| Test | Asserts | Then |
|---|---|---|
| `design_8_a_parameter_divergence_ends_the_stream_with_a_null_loss` | the DESIGN.md § 8 sentence, `lr = 1e38` parameter fixture under `--serve` | red: loss 9.693204879760742, not null |
| `the_loss_of_the_final_evaluation_is_checked` | failure at the last step: exit 1, last line `step k loss inf`, no `param` lines, no final checkpoint | green |
| `no_checkpoint_holds_a_non_finite_parameter` | `--checkpoint-every 1` writes no non-finite checkpoint and none past the failing step | green; catches "save before the check" |
| `a_resumed_run_fails_at_the_same_step` | resume from `c-(k-1)` names the same step | green |
| `a_diverged_serve_run_exits_1` | SIGTERM after divergence: exit 1, last event `RunFailed`, previous step `loss: null` | green |

2. **Suite**: `cargo test --workspace --no-fail-fast`, EXIT=101: 917 passed, 1 failed, 47 ignored; the one failure is the red case above.

3. **Implementor's three choices**
   - Edited `observatory_signal_adversary.rs`: a contract update, not a weakening; the story says "the stream ends after it". Note, inferred: the 50- and 120-step runs may now stop at one early step, so the "last step is a 50th" branch may go unexercised.
   - Diagnostic format: the story's own "(in the existing diagnostic format)" binds `E-XXX: message` (DESIGN.md § 8.1, `diag.rs:102`); the story's literal `error[E-TRAIN-001]` is the text to amend.
   - Golden from the base binary: matches `website/data/modelspec.terminal.json:11` and `website/docs/getting-started.md` at the base.

4. **Attacked, not broken**: NaN and ±inf; failure at step 0 and at the last step; a checkpoint at the diverging step; resume; `--serve` order and exit 1; stdout against stderr; exit codes; parameter ids.

5. **Coordinator action** (dee2bad): DESIGN.md § 8 and the CHANGELOG line now say the loss is `null` when the loss is the non-finite value; the red case asserts the finite loss of a parameter divergence and is green. The story text is amended to the existing format. The Vulkan check is filed as a draft story.

```findings
- file: DESIGN.md
  line: 519
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: DESIGN.md and CHANGELOG.md:14 promise a null loss in the last StepCompleted of every E-TRAIN-001 run under --serve, but a parameter divergence correctly streams the finite loss of that step (adversary_train_non_finite.rs:282 red).
- file: crates/modelspec-runtime/src/vulkan/runner.rs
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: The Vulkan runner the story's out-of-scope line waits for already exists and does no non-finite check, so a diverging --backend vulkan run should still exit 0; not run, since the GPU was off limits.
```
