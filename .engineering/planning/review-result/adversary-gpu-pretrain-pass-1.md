---
format: aep.planning-md/3
id: review-result:adversary-gpu-pretrain-pass-1
kind: review-result
status: active
title: Adversary pass 1 on the BERT-tiny GPU pre-training run and its record
relations:
- reviews: story:bert-tiny-gpu-pretraining
revision: 1
---
unit: story:bert-tiny-gpu-pretraining (worktree wt-6e5ecc4fcb2d, base 7e371a1 plus pass B's edits)
verdict: findings 3 (blocker 0, warning 2, note 1); 2 red cases (F1, F2)
cases: executed 1910→1915 (5 page cases in the default suites), plus 4 record cases and 1 device case run per `--test` with the run directory in `MODELSPEC_GPU_PRETRAIN_RUN`
origin: introduced 2 / pre-existing 1 / undecided 0
wrote-outside-worktree: `<wave-root>/gpu-pretrain/adv-scratch/` (suite log, control run stdout and validation log); `target/tmp/adversary_gpu_pretrain/` (probe spec, two event files, two validation logs)
needs-coordinator: the status row names `crates/modelspec-cli/tests/adversary_gpu_pretrain_cli.rs::run_record_page_agrees_with_the_committed_run_lines`; F1 and F2 go back to pass B as page edits

## Cases added

| file | cases | now |
|---|---|---|
| `crates/modelspec-cli/tests/adversary_gpu_pretrain_cli.rs` | 5 in the default suites: the page's 21 eval rows, 201 step losses, final loss and accuracy, margin (0.1590385 nats, 2.30 %), first evaluation below the baseline, step count, 2,609,690 bytes / 10,003 lines, SHA-256, last eval loss below 6.926537926202018, wall and eval-time estimate against the committed fixture; the page against `examples/bert_tiny.mspec` and the corpus table; the figures across README, AGENTS, index, roadmap, CHANGELOG and the pitch; F1 (rounding of the last-1,000-step gain); F2 (the baseline over the evaluated rows). 4 record: the fixture is the run's stdout; every event valid against the schema, sequences 0..10002 without a gap, one StepCompleted per step 0..10000, every step and eval line equal to its event as f32 text, the 9,980 other steps without a metric; the replay streams 10,003 frames and exits 0 on SIGTERM; every accuracy a whole number of hits out of 41,857 and the last loss below the unigram loss of the evaluated rows. 1 device: two 100-step runs byte-identical apart from ids, with the `data`, 41 `init`, step 0 and eval 0 lines of the full run (165 s) | 3 green, 2 red (F1, F2); 4 green; 1 green |
| `crates/modelspec-cli/tests/fixtures/bert_tiny_gpu_pretrain/stdout-lines.txt` | the 223 `data`, `step` and `eval` lines of the run's stdout, 5,996 bytes | |

## Suite run after the cases existed

| command | result | exit |
|---|---|---|
| `cargo test --workspace --no-fail-fast` | 278 suites, 1913 passed, 2 failed (F1, F2), 194 ignored | 101 |
| `task docs-check` | current | 0 |
| `cargo test -p modelspec-docs` | passed | 0 |
| `cargo fmt --all --check` | | 0 |
| `cargo clippy -p modelspec-cli --tests -D warnings` | clean | 0 |

## Findings

| # | file:line | verdict | origin | measured | reaches |
|---|---|---|---|---|---|
| F1 | `website/docs/runs/bert-tiny-gpu-pretraining.md:153` | NEEDS-CHANGE (warning) | introduced | the gain of the last 1,000 steps is 6.7717495 − 6.7674994 = 0.0042501, which rounds to 0.0043; the page truncates it to 0.0042 | the public page |
| F2 | `website/docs/runs/bert-tiny-gpu-pretraining.md:15` (also :30, :32, :151) | NEEDS-CHANGE (warning) | introduced | `packed.json`'s baseline 6.926537926202018 is the add-one unigram loss over every non-special HeldOut token (`dataset/mlm.rs:497-507`); the eval loss covers only the 41,857 selected rows of the 68 full batches (`train/eval.rs`); over those rows the unigram loss is 6.904154117063165, so the margin is 0.1366547 nats (1.98 %) and the first evaluation below it is step 2,500, not 1,500; scenario 1 holds on both populations (6.7674994 < 6.9041541) | the page's baseline sentence, margin and "first below" step |
| F3 | `crates/modelspec-runtime/src/vulkan/mod.rs:299` | INFEASIBLE (note) | pre-existing | `take_validation_messages` has no caller outside tests, so `train` under `MODELSPEC_VK_VALIDATION=1` collects messages and never reports them; the run used `=0`, so the record carries no validation evidence; a forced layer log of the 100-step probe held only `CURRENT-VALIDATION-ENABLED` | any CLI device run; the draft `story:cli-fails-on-vulkan-validation-messages` |

## Attacked and held

- Scenario 1: `EXIT=0`; stderr is the `serving` and `ui` lines only; 6.7674994 < 6.926537926202018; the device UUID of `inspect-device` matches the RTX 3090 in `/proc` and `nvidia-smi`; `driver_version` 2559459584 decodes to 610.57.04.
- Scenario 2: the eval loss falls strictly at every one of the 21 evaluations.
- Scenario 3: 10,003 valid events, sequences 0..10002, one StepCompleted per step 0..10000, all 201 step losses and 21 eval pairs equal their events as f32 text, the other 9,980 steps carry no metric; the replay streams 10,003 frames equal to the file's lines and exits 0 on SIGTERM.
- Scenario 4: the page carries the commands, commit, version, device, baseline, final loss and accuracy, margin, wall, event-file size and SHA-256 and the replay command; every figure recomputes from the files except F1 and F2.
- The accuracy plateau: every accuracy is a whole number of hits out of 41,857 (8 at step 0, 2,239 from step 1,000 to 3,500, 3,178 at the end); 2,239 is the count of token 2006, the most frequent token in Train and among the held-out targets, so the model predicted one token for every mask on the plateau.
- Reproducibility: two 100-step runs give identical stdout and events apart from ids; their `data`, 41 `init`, step 0 and eval 0 lines equal the full run's; evals 9.029848 / 8.713591 / 8.539593 as in unit 20.
- The baseline recomputes to the bit; the retrained tokenizer has the same SHA-256; the binary was built 29 s after 7e371a1 and reports 0.18.0; no `/home/` path in tracked files.

## Findings block

```findings
- file: website/docs/runs/bert-tiny-gpu-pretraining.md
  line: 153
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the gain of the last 1,000 steps is 6.7717495 - 6.7674994 = 0.0042501, which rounds to 0.0043, while the page truncates it to 0.0042
- file: website/docs/runs/bert-tiny-gpu-pretraining.md
  line: 15
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the page compares the eval loss over the 41,857 evaluated masked rows with packed.json's baseline over every non-special HeldOut token; on the evaluated rows the unigram loss is 6.9041541, so the margin is 0.1366547 nats (1.98 %) and the first evaluation below it is step 2,500, not 1,500
- file: crates/modelspec-runtime/src/vulkan/mod.rs
  line: 299
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: pre-existing
  message: train never reports the validation messages it collects under MODELSPEC_VK_VALIDATION=1, and the run used =0, so the run record carries no validation evidence; the device probe forced a layer log and found it clean at 100 steps
```
