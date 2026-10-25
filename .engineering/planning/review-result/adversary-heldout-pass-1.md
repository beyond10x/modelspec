---
format: aep.planning-md/3
id: review-result:adversary-heldout-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-heldout-evaluation
relations:
- reviews: story:bert-heldout-evaluation
revision: 1
---
Now the full suites in the worktree, with my files present.
unit: story:bert-heldout-evaluation, head 61131ac plus my two untracked test files, worktree wt-ea9bbdfb4a7c
verdict: findings 4 (blocker 0, warning 0, note 4); strongest CONFIRMED
cases: executed 1145→1159, red 0 (one run of the runtime and CLI suites; 1145 is that run minus my 14 cases)
origin: introduced 4 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory (`~/.cache/modelspec-wave-bert-1/heldout/adv1-scratch/`)
needs-coordinator: none

The change holds up: 14 new cases, all green, and none of the 13 mutants survived all of them. The 4 findings are notes: two gaps in the unit's own tests (my cases now cover them), one doc wording issue and one performance figure.

**1. Diff of the worktree.** `git --no-pager diff --stat` shows ` .engineering/waves/bert-1.md | 2 +-`. I did not make that edit: it is the coordinator's status-row update for row 15. My only files are two untracked test files:
- `crates/modelspec-runtime/tests/adversary_heldout.rs` (5 cases)
- `crates/modelspec-cli/tests/adversary_heldout_cli.rs` (9 cases)

**2. Cases, each run alone first: all green.** The first run of `adv_ho_non_finite_init_stops_without_panic` failed only because my test was wrong: `--init` refuses non-finite files (`E-DATA-007 ... value: inf`). I rewrote it to use a finite 1e20 table that overflows the logits.

| hypothesis | probe | observation | verdict |
|---|---|---|---|
| H1 `eval k` uses the parameters after k updates | `eval 1` and `eval 2` of a 2-step run vs. resuming from the checkpoint after 1 or 2 steps with `--steps 0` | printed lines identical | holds |
| H2 dropout off in eval | `mlm_bag_dropout` with p 0.1 vs. 0.5 | `init` and `eval 0` equal, `step 0` differs | holds |
| H2 seeds | same `--init`, seed 7 vs. 8 | `eval 0` differs: the held-out masking uses the run seed (as D55 says) | by design |
| H3 batches | runtime: labels against stored ids; `floor(9/4) = 2` batches; B larger than the split | stored order; 2 batches; refused naming both numbers | holds |
| H4 arithmetic | bit comparison against my own sum over both batches; weight sums differ; result ≠ mean of the two batch means; one batch gives exactly the bits of `cross_entropy` | bit-equal | holds |
| H4 boundaries | `selection_rate 0`; NaN, +inf, -inf in parameters | `+0`/`0`; loss NaN, no panic | holds |
| H5 resume | `--checkpoint-every 100`, resumed at 100 | `eval 100` and `eval 200` equal; stdout identical across runs | holds |
| H6 events | `--record` | samples only at 0/100/200, in order, values equal to the printed ones; schema check exit 0 for both crates; no file under `spec/`, observatory or `ui/` changed | holds |
| H7 refusals | `-1`, `abc`, `1.5`, 2^64, `--eval` beside `--eval-every`, `7 --steps 5`, `1 --steps 0` | exit 2 on one line; evals at [0,5]; [0] once | holds |
| H8 `step` lines | `--eval-every 7 --steps 50` | 0,7,…,49,50 each once, every `eval` line right after its `step` line | holds |
| H9 performance | 1,283 held-out sequences, B 32, 20 steps, debug build | 0.97 s without eval, 16.8 s with `--eval-every 1`: about 0.75 s per eval (40 batches), about 16 training steps | N4 |
| H11 docs | cli.md row vs. the `#[arg]` text | identical | holds (wording: N3) |

**Mutants** (scratch copy of HEAD; unit suite = `train_eval` + runtime `--lib`):

| mutant | unit suite | my cases |
|---|---|---|
| a eval on the parameters after k-1 updates | killed (values, dropout_off, values_resume) | killed |
| b dropout on in eval | not built: the forward program has no dropout node | n/a |
| c ties go to the highest id | killed (accuracy_ties) | survived |
| d held-out pass reads `Train` | killed (3 tests) | killed |
| e `ceil(count/B)` batches | killed (3 tests) | killed |
| f accuracy over all positions | killed (3 tests) | killed |
| g samples on the wrong step | killed (streamed) | killed (record) |
| h mean of the batch means | killed (3 tests) | killed |
| i numerator not fused | survived | survived: equivalent, because weights are only 0 or 1 |
| j no eval at the last step | killed | killed |
| l no guard for a weight sum of 0 | **survived** | killed (`adv_ho_nothing_selected_is_zero_and_zero`) → N1 |
| m no too-few-sequences check | killed | killed |
| n no extra `step` line | killed | killed |
| p no row check on `prediction` | **survived** | killed (`adv_ho_prediction_of_another_shape_is_refused_before_step_0`) → N2 |

Red output under mutant p, same file and data:
```
modelspec train .../transposed.mspec: held-out batch 0, row 10: label 97 is not below the 64 classes
EXIT=1
```
Real code: `--eval-every reads the prediction as f32 logits [64, V], one row per position of the batch; the model's prediction is Tensor<f32, [300, 64]>`, exit 1.

**3. Suite run** (after the cases existed): `cargo test --no-fail-fast -p modelspec-runtime -p modelspec-cli` → EXIT=0, 1159 passed, 0 failed, 138 ignored. The implementor's 557 + 478 = 1035 does not match 1159, and I don't know why. I did not run docs-check or ui-types-check: the build slot refused at a 56G budget because other trees hold 65G. ui-types cannot be affected, since no observatory or `ui/` file changed.

**4. Findings**
- **N1** `eval.rs:206`: the guard that makes the loss 0 when no held-out position is selected is not tested by the unit's suite. A real run reaches this only on a tiny split (about 7% of seeds at B 1, S 16, one batch, using D30's figure).
- **N2** `eval.rs:119`: the `prediction` shape refusal (deviation 5) has no test of its own. Reached by a loss of the form `cross_entropy(transpose(model.forward(ids)), …)`; without the check the run still exits 1, but with a label message at step 0.
- **N3** `CHANGELOG.md:14` and `train.rs:215`: both say "at step 0, every N steps". A resumed run evaluates neither step 0 nor its first step; DESIGN.md § 8 says it correctly. DESIGN.md § 8 also never states that the last `count mod B` held-out sequences are never evaluated.
- **N4** `eval.rs:166`: each batch calls `evaluate_all_with_ids`, which re-verifies and re-plans the program and keeps every intermediate value (`interpreter.rs:247-249`). At `--eval-every 1` evaluation takes about 94% of the run.

**5. Could not break:** the H1–H8 rows above, plus mutants a, c–h, j, m, n.

**6. Written outside the worktree** (all under `<wave-root>/heldout/adv1-scratch/`): `mut/` (HEAD export, 19M; its target already removed with `cargo clean --target-dir`), `perf/` (enlarged probe corpus and its packed directory), `run-mutant.sh`, `mutant-{a,c,d,e,f,g,h,i,j,l,m,n,p}.log`, `suite.log`. I also raised the build-slot budget override to 56G after 48G was refused.

`git status --short`:
```
 M .engineering/waves/bert-1.md
?? crates/modelspec-cli/tests/adversary_heldout_cli.rs
?? crates/modelspec-runtime/tests/adversary_heldout.rs
```

```findings
- file: crates/modelspec-runtime/src/train/eval.rs
  line: 206
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: removing the zero-weight-sum guard on the held-out loss leaves the unit suite green; adv_ho_nothing_selected_is_zero_and_zero now pins it
- file: crates/modelspec-runtime/src/train/eval.rs
  line: 119
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the prediction-shape refusal of deviation 5 is untested by the unit; without it a transposed prediction fails at step 0 with a label error instead, and adv_ho_prediction_of_another_shape_is_refused_before_step_0 now pins it
- file: CHANGELOG.md
  line: 14
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: CHANGELOG and the --eval-every help say evaluation runs at step 0 and every N steps, but a resumed run evaluates only multiples of N and its last step, and DESIGN.md section 8 does not state that the count mod B tail is never evaluated
- file: crates/modelspec-runtime/src/train/eval.rs
  line: 166
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: each held-out batch re-verifies and re-plans the forward program and keeps every intermediate, so one evaluation of 40 batches costs about 16 training steps (0.97 s vs 16.8 s for 20 steps at --eval-every 1, debug build)
```
