---
format: aep.planning-md/3
id: review-result:adversary-muon-pass-1
kind: review-result
status: active
title: Adversary pass 1 on muon-optimizer
relations:
- reviews: story:muon-optimizer
revision: 1
---
unit: story:muon-optimizer (worktree wt-c49c0aa4d15d, head c6f5970)
verdict: findings 4 (blocker 0, warning 0, note 4); strongest verdict CONFIRMED
cases: executed 1598→1618, red 4
origin: introduced 4 / pre-existing 0 / undecided 0
wrote-outside-worktree: `<wave-root>/muon/adv1-scratch/` (probe, corpus, mutant copy, logs; the mutant target cleaned)
needs-coordinator: ADV-MUON-1, the exit code: acceptance 6 and D67 (f) say 2, the code and DESIGN.md § 8 say 1, and no decision records the change

## Cases added

| file | cases | now |
|---|---|---|
| `crates/modelspec-runtime/tests/adversary_muon.rs` | 13: a tall `[4,3]` and a wide `[3,4]` weight over 12 steps against an independent f32 reference, bit for bit, including the final `b`; two Muon matrices with their own scales; shapes `[1,1]`, `[1,16]`, `[16,1]` at `ns_steps` 1 and 10; a zero gradient; a rank-1 gradient; a `[64,32]` weight with values from 1e-3 to 1e3; the schedule's edges; `--steps` override; determinism; a file with no matrix bit-equal to plain `adamw` | 11 green, 2 ignored and red |
| `crates/modelspec-core/tests/adversary_muon.rs` | 5: the split rule (table also read by matmul, tied decoder, rank 3, unused matrix), each scale rounded once from f64, transpose count, printer round trip | 5 green |
| `crates/modelspec-cli/tests/adversary_muon_cli.rs` | 6: 17 boundary edits with code and column, `muon()`, two refusals reported at once, `compile` and `infer` paths, `--steps` at the warm-up, the exit code, other seeds | 4 green, 2 ignored and red |

## Red cases (verbatim)

ADV-MUON-3 `adv_muon_3_reported_rate_without_a_matrix_is_the_applied_one`:
```
assertion `left == right` failed: reported rates (Muon lr = 0.02) against the rates the AdamW parameters apply (adam_lr = 0.01)
  left: [0.01, 0.02, 0.0175, 0.015, 0.012499999, 0.01, 0.0075, 0.005, 0.0025, 0.0]
 right: [0.005, 0.01, 0.00875, 0.0075, 0.0062499996, 0.005, 0.00375, 0.0025, 0.00125, 0.0]
```
ADV-MUON-4 `adv_muon_4_an_overflowing_norm_is_not_silent`:
```
assertion `left != right` failed: step 1 is the decay alone: the largest gradient element is 9.522655e29, the losses [1.453975e30, 1.4501327e30, 1.4475815e30, 1.4463094e30, 1.4463094e30], the final buffer finite: true
```
ADV-MUON-1 `adv_muon_1_vulkan_train_exits_2_as_acceptance_6_says`:
```
assertion `left == right` failed: E-TARGET-011: optimizer `muon` has no Vulkan kernel yet
  left: Some(1)
 right: Some(2)
```
ADV-MUON-2 `adv_muon_2_heldout_comparison_holds_under_other_seeds`:
```
seed 2: muon 5.944862 adamw 5.9250245
seeds where Muon's held-out loss at 300 is above AdamW's: [(2, 5.944862, 5.9250245)]
```
Across seeds 1 to 10, Muon is lower in 9. The seed-7 margin is 0.0038.

## Suite run after the cases existed

| crate | exit | passed | failed | ignored | before |
|---|---|---|---|---|---|
| modelspec-core | 0 | 497 | 0 | 3 | 492 |
| modelspec-runtime | 0 | 597 | 0 | 12 | 586 |
| modelspec-cli | 0 | 524 | 0 | 46 | 520 |

fmt and clippy `-D warnings` on the three files exit 0.

## Mutants (scratch copy)

| mutant | killed by the unit's tests | killed by the adversary's cases |
|---|---|---|
| a Nesterov dropped | `muon.rs` one_step, newton_schulz_singular_values | 6 runtime |
| b coefficients permuted | same two | 6 runtime |
| c `ns_steps − 1` | same two, plus train_muon ir and held-out tests | core scale/transpose test, 6 runtime |
| d transpose the other way | one_step, newton_schulz_singular_values | 6 runtime |
| e `sqrt(min)` | one_step, newton_schulz_singular_values | core scale test, 6 runtime |
| f decay coupled | one_step | 7 runtime |
| g norm epsilon dropped | newton_schulz_singular_values only | zero_gradient, two matrices, degenerate shapes |
| h embedding-read matrix given to Muon | train_muon: both ir tests | core: both split tests |

None survived.

## Findings

| id | file:line | verdict | origin | what was measured | what reaches it |
|---|---|---|---|---|---|
| ADV-MUON-1 | crates/modelspec-cli/tests/train_muon.rs:263 | CONFIRMED | introduced | `train --backend vulkan` exits 1; acceptance 6 and D67 (f) say 2 | any Muon file with `--backend vulkan`; the 1 matches AdamW and AGENTS.md's exit codes |
| ADV-MUON-2 | crates/modelspec-cli/tests/adversary_muon_cli.rs:381 | CONFIRMED | introduced | acceptance 5's `muon <= adamw` fails at seed 2; the seed-7 margin is smaller than the spread between seeds | changing `seed` in `mlm_dense.mspec`; DESIGN.md already says "not worse, not better" |
| ADV-MUON-3 | crates/modelspec-runtime/src/train/optimizer.rs:352 | CONFIRMED | introduced | with no Muon parameter, the run reports the schedule of `lr`, a rate no parameter applies | a `muon(...)` file with only vectors or embedding tables; `Observed::learning_rate` doc (runner.rs:254) still says "rate of `adamw`" |
| ADV-MUON-4 | crates/modelspec-core/src/ir/training.rs:1160 | INFEASIBLE | introduced | gradient near 1e30: `sum(u*u)` overflows to inf, `x = 0`, the weight gets weight decay only, no diagnostic; AdamW on the same file stops with E-TRAIN-001 | nothing found; constructed (prediction times 1e15); torch's f32 norm overflows the same way |

## Attacked, not broken

Bit-equality beyond step 3 and of `b`; the wide (no-transpose) branch; the reduction order of the sum of squares over a wide value range; each scale rounded once from f64; zero gradient, rank-1 gradient, degenerate shapes at `ns_steps` 1 and 10; the split rule in four cases; schedule edges, `warmup_steps = 0`, `--steps` below, at and above the warm-up; determinism, AdamW unchanged when Muon is present; 17 boundary edits; `compile`, `infer`, `ir --stage kernel`; `--record` then `observe` (the observatory has no optimizer kind); the opponent on the pinned venv reproduces 6.6212 / 6.7239 with the same split and no `/home/` path; diagnostics, concept page, status row, CHANGELOG and `MuonConfig`.

```findings
- file: crates/modelspec-cli/tests/train_muon.rs
  line: 263
  category: acceptance
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: train --backend vulkan on a Muon file exits 1 where acceptance 6 and D67 (f) say exit 2, and no coordinator decision records the change
- file: crates/modelspec-cli/tests/adversary_muon_cli.rs
  line: 381
  category: property
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: acceptance 5's muon <= adamw at step 300 holds at seed 7 by 0.0038 nats but fails at seed 2, so the quoted comparison is within seed noise
- file: crates/modelspec-runtime/src/train/optimizer.rs
  line: 352
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: a muon run with no Muon parameter reports the schedule of lr as its learning rate although every parameter applies adam_lr's schedule
- file: crates/modelspec-core/src/ir/training.rs
  line: 1160
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a gradient whose squares overflow f32 makes the Frobenius norm infinite and the Muon update silently weight decay only, with no E-TRAIN-001, in a constructed state no workflow was shown to reach
```
