---
format: aep.planning-md/3
id: review-result:adversary-bert-vs-pytorch-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-tiny-vs-pytorch-cuda
relations:
- reviews: story:bert-tiny-vs-pytorch-cuda
revision: 1
---
The suite is running in the background; I'll wait for it to finish.
unit: story:bert-tiny-vs-pytorch-cuda (adversary pass 1), commit 34dee95 (pass A, committed by the coordinator during this pass; tree clean apart from my file)
verdict: blockers 1, warnings 3, notes 2
tests: crates/modelspec-docs/tests/adversary_bert_vs_pytorch_page.rs, red 3 (adversary_adv_bp_4…, adversary_adv_bp_5…, adversary_adv_bp_6…; failure lines below)
wrote-outside-worktree: 1 directory, <wave-root>/bert-vs-pytorch/adv-scratch/ (part 6)

**1. `git --no-pager diff --stat`:** empty. My only addition is an untracked test file, and `git status --short` is at the end. My probe imported the opponent, which left an `opponents/bert_tiny_pretrain/__pycache__/` (written 10:39:56). I deleted it, so no non-test path remains.

**2. Cases added.** All three red cases fail on their assertion, not on a compile error. The control case, the same row at `MatmulOnly` 37,446,746,112 on the 100-step grid, is green.

Run of the file alone (first run of anything I wrote):
```
test adversary_adv_bp_4_a_bert_row_counted_by_all_sites_is_refused ... FAILED
test adversary_the_control_bert_row_is_accepted ... ok
test adversary_adv_bp_5_a_bert_row_whose_steps_to_target_was_never_evaluated_is_refused ... FAILED
panicked at crates/modelspec-docs/tests/adversary_bert_vs_pytorch_page.rs:108:5:
assertion `left != right` failed: a bert-tiny-pretrain row counted by AllSites, 39,400,695,809 FLOP and not 1,170,210,816·32 = 37,446,746,112, was accepted
  left: Some(0)
 right: Some(0)
panicked at crates/modelspec-docs/tests/adversary_bert_vs_pytorch_page.rs:135:5:
assertion `left != right` failed: steps_to_target 3450 with eval_every 100, a step no evaluation ran at, was accepted
  left: Some(0)
 right: Some(0)
test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 14.23s
```
ADV-BP-6 run alone, after it was added:
```
panicked at crates/modelspec-docs/tests/adversary_bert_vs_pytorch_page.rs:150:5:
target_loss calls the unigram baseline 6.926537926202018 the reference run's final held-out loss (that is 6.7674994): The held-out loss a time-to-target run trains to, such as the `bert-tiny-pretrain` reference run's final held-out loss `6.926537926202018`; with steps_to_target and seconds_to_target, required on a `bert-tiny-pretrain` run.
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s
```

**3. Suite run (after the cases).** `CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_INCREMENTAL=0 B10X_BUILD_REPO_BUDGET_G=90 b10x-build-slot cargo test --workspace --features vulkan --no-fail-fast`
```
error: 1 target failed:
    `-p modelspec-docs --test adversary_bert_vs_pytorch_page`
EXIT=101
```
Summed `test result` lines: passed 1973, failed 3, ignored 199. Cases executed went from 1,972 (pass A's reported count) to 1,976. An earlier run without `--no-fail-fast` stopped at the first failing target, and I discarded its count.

**4. Findings**

| id | sev | verdict / origin | claim | evidence | expected fix |
|---|---|---|---|---|---|
| ADV-BP-1 | blocker | NEEDS-CHANGE / introduced | The two sides' µs per step (and so tokens/s and GFLOP/s) use different statistics. The opponent reports the median over steps (`bert_tiny_pretrain.py:693`). `--clock` gives ModelSpec only a sum, and the draft report divides it by the steps, which is a mean. The page text (`benchmarks.rs:1834`) says "median of the timed run". The block opponent uses span ÷ steps. Reached by the coordinator's timed commands and the draft's Results note. | Unchanged opponent, `--steps 300 --warmup-steps 20 --eval-every 100 --checkpoint …/scratch/step0.msckpt`. Eager: `"median_us_per_step": 8498.63`, `"train_seconds": 2.85673` (mean 9,522 µs, **+12.0 %**). Compile: 4186.54 against 1.33467 s (mean 4,449 µs, **+6.3 %**). Pass A's own smoke: 7253 against 1.596 s/200 (+10 %). | The opponent reports `train_seconds/steps` per run and the median of 3 runs, as `transformer_block.py` does. Or ModelSpec prints a per-step median and the opponent keeps its median. Either way the page text matches what is reported. |
| ADV-BP-2 | note | CONFIRMED / introduced | The opponent's clock charges the pageable host-to-device copy and the scheduler step. ModelSpec's clock charges only the device-side copy from a staging buffer already written. The opponent's first timed step also carries AdamW's first-use cost now and then, which its warm-up does not cover. This does not move the ratio: ModelSpec is about 236 ms per step in release (`train_seconds 47.254723 … steps 200`). | Probe `adv-scratch/opp_clock_parts.py`, 200 eager steps: `h2d median_us 91.6`, `sched median_us 23.2`, `step median_us 8102.1 … first_us 146741.3`. That is about 1.4 % of the opponent's own step. Dropout-free run: steps 0–50 took 0.763 s, steps 50–100 took 0.417 s. The unchanged 300-step run showed no such excess. | Pin and stage the batch before the clock starts, and include one AdamW step in the warm-up, on a discarded optimizer. |
| ADV-BP-3 | note | CONFIRMED / introduced | A chunked Vulkan run sets `training: elapsed` (`runner.rs:854`). That span includes the final loss evaluation and checkpoint writes (RunStats doc `runner.rs:108`). The `--clock` help (`train.rs:276-280`) says both are excluded. The BERT figure does not reach this: it runs stepwise. | Read from the code only; no run. | Subtract them, or say so in the help and the module doc. |
| ADV-BP-4 | warning | CONFIRMED / introduced | A `bert-tiny-pretrain` row counted as `AllSites` skips the `1,170,210,816·B` check (`benchmarks.rs:397`). Its count is published beside PyTorch's `MatmulOnly`, against the acceptance's "the same FLOP count for all three". | `adversary_adv_bp_4…`, red (above) | `bert_problems` requires `MatmulOnly` with the formula's count. |
| ADV-BP-5 | warning | CONFIRMED / introduced | `steps_to_target` is not checked against `eval_every` (`benchmarks.rs:263`). A crossing at a step that was never evaluated (3450 at an `eval_every` of 100) gets published. | `adversary_adv_bp_5…`, red | Refuse when `steps_to_target` is not a multiple of `eval_every`. |
| ADV-BP-6 | warning | CONFIRMED / introduced | The `target_loss` summary (`spec/domains/benchmark.yaml:98`, and the generated schemas) calls 6.926537926202018 the reference run's final held-out loss. It is the `packed.json` unigram baseline; the final loss was 6.7674994. Recording the final loss instead would time each side to about step 10,000 instead of about step 1,500. | `adversary_adv_bp_6…`, red | Reword it as the held-out unigram baseline and regenerate the schemas. |

**5. Attacked and could not break**
- **Stepwise clock:** it sums each update from recording the command buffer to the end of its host wait. It excludes the eval batches, the final loss, the observer and the host draw. The value on the eval line at step k is the clock through k updates. Release smoke: 47.25 s train plus 97.93 s eval, inside 151.07 s of wall time.
- **CPU clock:** the eval at step k reads the time set after update k, and the observer's and checkpoints' time is subtracted. `--steps 0` gives a train time of 0. A run that diverges prints no clock line, as documented.
- **`--clock` off:** the format code is byte-identical to before for both the `eval` lines and the `param` lines.
- **Opponent fidelity:** both sides run with dropout set to 0 (copies `adv-scratch/bert_tiny_w20_p0.mspec` and `opp_nodrop.py`) on the same step-0 checkpoint, 200 steps, warm-up 20. Held-out losses, ModelSpec / PyTorch:
  - step 50: 8.686357 / 8.686356937953352
  - step 100: 8.26966 / 8.26966160145777
  - step 150: 8.069926 / 8.06992955280392
  - step 200: 8.011226 / 8.011230499349054

  So the AdamW update, schedule, batches, held-out set and initial parameters are the same on both sides. Pass A's gap of about 0.001 is dropout noise.
- **FLOP rule:** I derived it independently from `examples/bert_tiny.mspec`; it equals `N·(54H²+24HF+6HV+24SH)`, 37,446,746,112 per step, and the opponent prints the same number.
- **Page refusals:** the refusals for missing target figures and wrong tokens per step work. A run that never reaches its target is refused, because it cannot give `steps_to_target`. A PyTorch row without `modelspec_version` is accepted, which is what the schema intends (it uses `opponent_version`). "Not measured" drops BERT once a row exists.
- **`--steps 0` checkpoint:** an AdamW resume is refused at the CLI ("the argument '--resume' cannot be used with the `adamw` optimizer …"), so no resumed run can drift from a fresh one.
- **`profile step_total`:** hand counts match the profile rows (CrossEntropy loss 4096·(4·8192+6)+1 = 134,242,305; CrossEntropyGrad 4096·(9·8192+3) = 302,002,176), and `uncounted_sites 0` holds. On the stepwise path the profile prints `steady_us_per_step unavailable` and `gflop_per_s unavailable`.

**6. Paths written outside the worktree:** `<wave-root>/bert-vs-pytorch/adv-scratch/` holding:
- the two `.mspec` copies;
- `opp_nodrop.py`, `opp_clock_parts.py`;
- `p0-step0.msckpt` and its `.out`/`.err`;
- `ms-p0.*`, `ms-w20-rel.*`, `opp-p0.*`, `opp-eager-300.*`, `opp-compile-300.*`, `cpu0.out`;
- `suite.log`, `suite-nff.log`.

About 82 MB in all. Inside the worktree but not visible to git: the test sandboxes in `target/tmp/adversary-bert-vs-pytorch-page/`, about 74 MB.

```findings
- file: opponents/bert_tiny_pretrain/bert_tiny_pretrain.py
  line: 693
  category: acceptance
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the opponent reports a per-step median while ModelSpec's clock yields only a mean, measured 6.3-12.0 % apart on the opponent itself
- file: opponents/bert_tiny_pretrain/bert_tiny_pretrain.py
  line: 677
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the opponent's clock charges a pageable host-to-device copy (91.6 us) and the scheduler step (23.2 us), and now and then AdamW's first-use cost, which ModelSpec's clock does not charge
- file: crates/modelspec-runtime/src/vulkan/runner.rs
  line: 854
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: a chunked run's train_seconds includes the final loss evaluation and checkpoint writes that the --clock help says it excludes
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 397
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: a bert-tiny-pretrain row counted as AllSites skips the 1,170,210,816·B check and is published beside MatmulOnly rows
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 263
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: steps_to_target is not checked against eval_every, so a crossing at a step that was never evaluated is published
- file: spec/domains/benchmark.yaml
  line: 98
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the target_loss summary calls the unigram baseline 6.926537926202018 the reference run's final held-out loss, which was 6.7674994
```

`git status --short`:
```
?? crates/modelspec-docs/tests/adversary_bert_vs_pytorch_page.rs
```
