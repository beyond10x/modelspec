---
format: aep.planning-md/3
id: review-result:adversary-eval-download-pass-1
kind: review-result
status: active
title: Adversary pass 1 on the per-batch held-out evaluation of the stepwise Vulkan run
relations:
- reviews: story:vulkan-eval-download-per-batch
revision: 1
---
unit: story:vulkan-eval-download-per-batch (worktree wt-9c2a4b3556de, unit commit 89d923c)
verdict: findings 3 (blocker 0, warning 1, note 2); no red case
cases: executed 2645→2653 (4 + 4 CPU cases in the default suites), plus 9 adversary device cases run per `--test`; red 0
origin: introduced 1 / pre-existing 1 / undecided 1
wrote-outside-worktree: `<wave-root>/eval-download/adv1-scratch/` (a source copy of head with its target cleaned, the base, head and head-release binaries, runs, logs, probe scripts)
needs-coordinator: none

## Cases added

| file | cases | now |
|---|---|---|
| `crates/modelspec-runtime/tests/adversary_eval_download.rs` | 4 CPU: `Tallying` fed batch by batch gives the bits of `evaluate` and `metrics_of`; partial tallies, extra batches and short or long logits are refused and a refused add does not advance the tally; a NaN row makes the loss NaN and counts as wrong. 2 device: `train_stepwise` in-process with sync validation on (a planted hazard first shows the validation is live), no message, host waits and submits 4 + 4·4, eval adds 4·4·P downloaded bytes and nothing to dispatches or barriers per step, training losses and final parameters the same bits with and without eval, twin runs repeat bit for bit | 6 green |
| `crates/modelspec-cli/tests/adversary_eval_download_cli.rs` | 7 device: one full batch plus a remainder; too few sequences gives the CPU's refusal; a resumed run evaluates 5 and 10; the record matches the printed eval lines; divergence at an evaluated step stops as the CPU does; 3 twin runs byte-identical; peak device bytes grow by one prediction (+117,120 against 4,723,200 for one per batch) | 7 green |

## Suite run after the cases existed

| command | result | exit |
|---|---|---|
| `cargo test --workspace` | 1910 passed, 0 failed | 0 |
| `cargo test -p modelspec-runtime --features vulkan` | 743 passed, 0 failed | 0 |
| clippy `-D warnings`, rustfmt on both files | clean | 0 |
| acceptance run (100-step copy of `examples/bert_tiny.mspec`, `--eval-every 50`, packed corpus) | exit 0, 101 s, eval 0/50/100 = 9.029848 / 8.713591 / 8.539593 | 0 |

## Hypotheses

| # | probe | observation | verdict |
|---|---|---|---|
| bits | base and head debug binaries, eval output byte for byte | twin and bag stdout identical on Vulkan and CPU (724,580 B for the twin on Vulkan) | holds |
| barriers | in-process sync validation, 3 runs | 0 messages, outputs identical; barrier mutants a and a2 survive everything (F2) | holds |
| edges | B = 5 with 1 batch, 0 batches, `--eval-every 1`, `--steps 0` | matches the CPU; waits 8, 5, 12, 16 as predicted; clean refusal | holds |
| resume | resumed at 4, `--eval-every 5` | evals 5 and 10 on both backends, waits 15 | holds |
| counters | the 100-step run | waits 305 = 101 + 3·68; bytes down = 4·101 + 12·params + 3·68·P | holds |
| memory | VmHWM sampled | 1,062,156 kB with eval against 1,061,880 kB without | holds |
| observer | `--record` | eval steps carry 2 metrics matching the printed text; other steps empty | holds |
| NaN | SGD at lr 1e30 | exit 1, E-TRAIN-001, same step and eval lists as the CPU | holds |
| docs | DESIGN § 8, CLI help, CHANGELOG, module doc | match the counts measured | holds |

## Mutants (scratch copy)

| mutant | unit suite | adversary cases |
|---|---|---|
| a: no barriers in `open` | survived | survived (F2) |
| a2: drop only ComputeToTransfer | survived | survived (F2) |
| b: batches in reverse order | killed (`vulkan_eval_per_batch`, `heldout_eval`) | killed |
| c: old `download_bytes` formula | killed (sizing test) | killed |
| c2: allocate batches·P at the call site | killed (logit length check) | killed |
| d: eval tally not reset | killed | killed |
| e: evaluate batches − 1 | killed | killed |
| g: drop the completeness guard in `Tallying::metrics` | survived | killed (F3) |
| e + g | killed by the wait count only | killed |
| v: planted WAW copy in the eval buffer | survived | killed (runtime device cases) |

## Findings

| id | file:line | severity / verdict / origin | what |
|---|---|---|---|
| F1 | `crates/modelspec-runtime/src/vulkan/mod.rs:299` | warning / CONFIRMED / pre-existing | the CLI never reads the validation layer's messages, so `MODELSPEC_VK_VALIDATION=1` on a CLI device test cannot fail on a validation or sync hazard (head and base alike) |
| F2 | `crates/modelspec-runtime/src/vulkan/stepwise.rs:746` | note / INFEASIBLE / undecided | the barriers `open` records between fenced submits are unobservable on this device and layer |
| F3 | `crates/modelspec-runtime/src/train/eval.rs:322` | note / CONFIRMED / introduced | the `Tallying::metrics` completeness guard had no case in the unit suite; the adversary's case now kills its removal |

## Attacked, not broken

Bits against base, ordering and determinism, edge counts, resume, counters, host memory, the recorded events, divergence, docs, the acceptance run.

```findings
- file: crates/modelspec-runtime/src/vulkan/mod.rs
  line: 299
  category: judgement
  severity: warning
  verdict: CONFIRMED
  origin: pre-existing
  message: the CLI never reads the validation layer's messages, so MODELSPEC_VK_VALIDATION=1 on a CLI device test, including vulkan_eval_per_batch, cannot fail on a validation or sync hazard
- file: crates/modelspec-runtime/src/vulkan/stepwise.rs
  line: 746
  category: mutant
  severity: note
  verdict: INFEASIBLE
  origin: undecided
  message: the barriers open() records between fenced submits are unobservable on this device and layer; removing them survives every case including in-process sync validation
- file: crates/modelspec-runtime/src/train/eval.rs
  line: 322
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the Tallying::metrics completeness guard had no case in the unit suite; adv_ed_tallying_refuses_a_partial_or_overlong_evaluation now kills its removal
```
