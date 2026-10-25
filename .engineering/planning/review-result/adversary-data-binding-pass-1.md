---
format: aep.planning-md/3
id: review-result:adversary-data-binding-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-train-data-binding
relations:
- reviews: story:bert-train-data-binding
revision: 1
---
Now I write the runtime test file.
unit: story:bert-train-data-binding, findings cover commit e5ea548 plus my two untracked test files (worktree wt-28897fda3bdc)
verdict: findings 5 (blocker 0, warning 3, note 2)
cases: executed 1660→1673, red 0
origin: introduced 5 / pre-existing 0 / undecided 0
wrote-outside-worktree: 4 paths
needs-coordinator: none

The implementation did not break: none of my 13 new cases fails on e5ea548. All five findings are about tests. The unit's own suite (`train_data`, 5 cases) caught only 3 of 11 planted mutants. My cases caught all 11.

**1. Diff.** `git --no-pager diff --stat` prints nothing because both new files are untracked. `git status --short`:
```
?? crates/modelspec-cli/tests/adversary_data_binding_cli.rs
?? crates/modelspec-runtime/tests/adversary_data_binding.rs
```
Both are test files. I changed no implementation file. Both pass `rustfmt --check` and `clippy -D warnings`.

**2. Cases added.** All green on e5ea548, so none is `#[ignore]`. The first run of each file had 1 red case, caused by my own mistake: I assumed more than 8 batches per epoch, but the fixture packs only 26 Train sequences, which is 6 batches of 4. I fixed the assumption; that was not a finding.

| case | asserts |
|---|---|
| runtime `adv_db_step_k_feeds_the_mlm_batch_of_step_k_by_name` | for steps 0, 1, 7, 6 and 7, all five tensors equal a direct `mlm_batch` call element for element, bound by name although declared in a scrambled order; `[64]` ids sit beside a `[4, 16]` key_mask |
| runtime `adv_db_another_seed_draws_another_batch` | seed 8 gives other `weights` than seed 7 at the same step |
| runtime `adv_db_run_on_reads_the_batch_of_the_absolute_step` | starting at 0, 1 and 5, the first loss is the forward pass on batch `start`, and the last loss is on batch `start+1` |
| runtime `adv_db_shapes_bind_and_refuse` | `[16,4]`, `[4,16,1]`, `[65]` and `[8,16]` key_mask are each refused with E-DATA-006 |
| runtime `adv_db_disagreeing_b_names_the_later_tensor_in_d48_order` | key_mask `[3,16]` declared before ids `[64]`: E-DATA-006 names key_mask and expects `[4, 16]` |
| cli `adv_db_binding_checks_refuse_naming_the_file` | binding id off by one digit, `seq_len` 0, edited masking rate, packed.json from another tokenizer, train.tokens 1 byte short: each exits 1 with one line naming the file |
| cli `adv_db_batch_of_all_train_sequences_and_one_more` | B=26 trains; B=27 exits 1 naming the directory and `Train` |
| cli `adv_db_dense_packed_decision` | dense names on a packed directory → E-DATA-005; a stray binding.json in a dense directory → E-DATA-005; a missing directory with dense names → not E-DATA-005, path named |
| cli `adv_db_data_line_and_run_started_only_on_packed_runs` | a dense run has no `data` line and no `data_binding_id`; a packed recording reads back with the binding's id |
| cli `adv_db_seed_reaches_the_batches_and_not_the_data_line` | with parameters fixed by `--init`, seed 7 and seed 8 print different step-0 losses but the same `data` line |
| cli `adv_db_resume_from_checkpoint_every_matches_the_unbroken_run` | resuming from `run-150.msckpt` gives the unbroken run's step-200 loss and parameters |
| cli `adv_db_all_five_tensors_train_from_the_cli` | a model with attention reading all five tensors trains, exit 0 |
| cli `adv_db_eval_beside_packed_data_is_refused_before_any_step` | exit 1, empty stdout |

**3. Suite.** `b10x-build-slot cargo test --workspace` in the worktree, run after the cases existed: 1673 passed, 0 failed, 141 ignored, `EXIT=0`. My binaries reported 8 and 5 passed; `train_data` reported 5 passed. The before count, 1660, is 1673 minus my 13.

**Mutants** (each planted in a scratch copy; each run of the unit's suite selected only `--test train_data`):

| mutant | unit suite | killed by my cases |
|---|---|---|
| a: feed batch `k-1` | survived | `run_on_reads_the_batch_of_the_absolute_step` |
| b: seed ignored for the batch | survived | `another_seed…`, `step_k…`, `seed_reaches_the_batches…` |
| c: any name accepted | killed (`refusals…`) | `dense_packed_decision` |
| d: bind by position | survived | 4 runtime cases |
| e: no packed.json ownership check | survived | `binding_checks…` |
| f: `data` line after `init` | killed (`trains…`, `resume…`) | `all_five…` |
| g: B above the Train sequence count allowed | survived (almost equivalent: `mlm_batch` refuses at step 0 anyway) | `batch_of_all…` |
| h: B taken from source order | survived | `disagreeing_b…` |
| i: binding id self-check removed | survived | `binding_checks…` (edited masking rate) |
| j: `[N]` length not checked | survived | `shapes_bind_and_refuse` |
| k: `data_binding_id` always None | killed (`trains_records…`) | `data_line_and_run_started…` |

**4. Findings** (all against e5ea548; "reached by" = how a user or the suite gets there):
- **F1, warning.** `train_data.rs:269`: the seed half of the reproducibility test cannot fail when the seed does not reach the batches, because the seed also changes the initial parameters (mutant b survived). Reached by: acceptance scenario 2.
- **F2, warning.** `train_data.rs:306`: the resume test cannot detect feeding a shifted step, because the whole run and the resumed run shift the same way (mutant a survived). Nothing in the unit's suite compares the fed tensors with `mlm_batch`. Reached by: the D48 resume rule.
- **F3, warning.** `train_data.rs:186`: no unit test checks binding by name. Binding by position feeds `labels`←positions and `weights`←key_mask, and the fixture still trains (mutant d survived).
- **F4, note.** `data.rs:775`, `:646`, `:855` and `:879`: the ownership check, the binding id self-check, the B order and the `[N]` length check are not guarded by the unit's suite (mutants e, i, h, j). My cases now guard them.
- **F5, note, a judgement call.** `data.rs:634` and `cli/src/train.rs:383`: an existing directory without binding.json goes the dense way, per D48. So `train mlm_bag --data <dataset-dir>` (a plausible mistake after `data prepare`), an empty directory, or `--eval <packed-dir>` all print `E-DATA-001: dense data needs exactly one input and one target`. That message names neither the directory nor binding.json, and DESIGN says E-DATA-001 does not apply to packed data. I observed this directly with the worktree binary.

**5. Attacked and could not break:**
- `[B·S]` and `[B, S]` carry the same row-major bytes.
- Batches wrap correctly across epochs.
- Resume and `--checkpoint-every` behave as D48 says.
- Every refusal exits 1 with one line, never 101; the order 005, 006, then 003 holds.
- The `data` line is printed on packed runs only and does not depend on the seed.
- `data_binding_id` is absent (not null) for dense and synthetic runs, and a recording that carries it reads back.
- `--data` with `--backend vulkan` is refused with exit 2.
- H6, speed: in release, B=24 S=16 (26 sequences cannot give B=32), 200 steps, best of 5: the packed run does 798 steps/s and the same model on one fixed batch does 809, so drawing the batches costs about 1.4% (6 µs per step for `mlm_batch`). `run_with_ids` allocates only the output list, not the output tensors. The allocation does not dominate, so there is no finding. A synthetic f32 matmul+mse run did 2834 steps/s, but it is a different model, so it is no comparison for the allocation.

**6. Paths outside the worktree:** in `<wave-root>/data-binding/adv1-scratch/` I left `mutants.log`, `suite.log`, `mutate.sh` and `perf_probe.rs` (216 KB). The scratch copy and its 2.2 GiB target were removed with `cargo clean --target-dir`. Inside the worktree, my tests left `target/tmp/adv_data_binding_*` directories, which I did not delete. The lease is released.

```findings
- file: crates/modelspec-cli/tests/train_data.rs
  line: 269
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the other-seed assertion passes when the batch ignores the seed because the seed also changes the initial parameters (mutant b survives)
- file: crates/modelspec-cli/tests/train_data.rs
  line: 306
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the resume test passes when step k feeds batch k-1 since both runs shift alike, and no unit test compares fed tensors with mlm_batch (mutant a survives)
- file: crates/modelspec-cli/tests/train_data.rs
  line: 186
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: binding declarations by position instead of by name still trains the fixture, so no unit test pins by-name binding (mutant d survives)
- file: crates/modelspec-runtime/src/train/data.rs
  line: 775
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the packed.json ownership check, the binding id self-check (646), the D48 B order (855) and the [B*S] length check (879) are each removable with the unit's suite green
- file: crates/modelspec-runtime/src/train/data.rs
  line: 634
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: an existing directory without binding.json under batch-tensor declarations, or --eval beside packed --data, is refused with E-DATA-001 naming neither the directory nor binding.json
```
