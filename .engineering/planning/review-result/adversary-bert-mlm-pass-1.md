---
format: aep.planning-md/3
id: review-result:adversary-bert-mlm-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-mlm-batches
relations:
- reviews: story:bert-mlm-batches
revision: 1
---
Formatting is clean. Now clippy on both crates and clean the mutant target.
unit: story:bert-mlm-batches, worktree wt-7f0603a76d88 at head 90b76e0 plus two untracked adversary test files
verdict: red
cases: executed 1301→1317, red 3
origin: introduced
wrote-outside-worktree: 5 paths, under <wave-root>/mlm/adv1-scratch/
needs-coordinator: story:bert-dropout still claims tag 3 (it is now 4); packed_id leaves out seq_len (a spec D26 decision); story:bert-mlm-cross-entropy must handle batches with nothing selected (sum(weights) = 0)

The code does what DESIGN.md § 2.3 says. Four mutants survive the unit's own tests, and three spec gaps are confirmed by red cases. None is a blocker.

**1. Diff.** `git --no-pager diff --stat` is empty. `git status --short` shows only my two files:
```
?? crates/modelspec-cli/tests/adversary_mlm_pack.rs
?? crates/modelspec-runtime/tests/adversary_mlm_batches.rs
```
Neither file contains a `/home/` path. `cargo fmt --all --check` and clippy with `-D warnings` on both crates exit 0.

**2. Cases** (each run alone before the suite)

Red, left in the tree as `#[ignore]` with the finding id:

| Case | Red output, verbatim |
|---|---|
| `adv_packed_id_changes_with_seq_len` (batches.rs:418), ADV-MLM-1 | `two PackedDataset records with seq_len 2 and 4 share packed_id  left: "e1c5be3e-e719-8610-b517-8221fa438956"  right: "e1c5be3e-e719-8610-b517-8221fa438956"` |
| `adv_load_refuses_a_vocabulary_size_the_spec_forbids` (batches.rs:278), ADV-MLM-4 | `panicked at crates/modelspec-runtime/src/dataset/mlm.rs:745:25: attempt to subtract with overflow` |
| `adv_records_breaking_the_spec_invariants_do_not_validate` (pack.rs:271), ADV-MLM-3 | `validate against the schema: ["PackedDataset seq_len 0", "PackedDataset vocabulary_size 5", "PackedDataset heldout_unigram_baseline_nats -1.0", "TrainingDataBinding seq_len 0"]` |

Green: 16 cases.
- **Runtime:** every row against the tag-3 derivation (B=4, the random token included); identical sequences at two rows draw differently; all-special sequences; V=6; epoch-order bijection (n = 0 to 1000, seed, epoch and epoch high word); n < B; a flipped byte refused by its hash; one-ULP binding_id; the same batches in two other processes; nothing-selected batches (71 of 1000 at S=16, B=1).
- **CLI:** `--seq-len` values `-1`, empty, `1.5`, `abc`, 2^64, `0x10` and ` 16` all exit 2; a missing heldout.raw exits 1 and leaves a non-empty `--out` byte-identical; a tokenizer_id mismatch exits 1; `--out` inside the dataset; the dropped tail is exactly the last `L mod S` ids; `data_binding_id` is optional.
- **Real release** (`adv_real_release_separators_and_baseline`, ignored, needs `ADV_MLM_REAL_PACKED`; passed in release): 28,471 `[SEP]` in train (28,472 − 1) and 59 in held-out (60 − 1). The baseline, recomputed independently, is 6.926537926202018, equal to the record. ln(8187) = 9.0103; train unigram entropy 6.946; perplexity 1,019, which is plausible. Selected share 0.1494 over 400 × 32 × 128 positions.

**Mutants.** I ran them on a scratch copy. The worktree was not touched.

| Mutant | Unit suite | Adversary suite |
|---|---|---|
| M1 select special tokens, M2 random token from [0,V), M3 tag 3 for the order, M4 stored order, M5 padded tail, M6 heldout hashed first, M10 no 2^63 offset, M12 smoothing over V | killed | — |
| M7 `row = p` (every sequence in a batch gets the same mask) | **survives** | killed |
| M8 random token from w1 instead of w2 | **survives** | killed |
| M9 hash check removed from `load` | **survives** | killed |
| M11 epoch high word dropped | **survives** | killed |

**3. Suite** (run after the cases existed): `b10x-build-slot cargo test --workspace` gave EXIT=0: 1317 passed, 0 failed, 114 ignored. The result lines printed 1319 because the two child-process runs echoed their own result line; their stdout now goes to null. "Before" is 1317 minus my 16 executed cases, taken from the same run, not a second run.

**4. Findings that are judgement calls**
- **ADV-MLM-5:** `.engineering/planning/story/bert-dropout.md:79` still says "Stream tag 3 is dropout". D26 and DESIGN.md § 2.3 now give tag 3 to masking and reserve tag 4 for dropout. That story's scenario at :148 asserts tag 3. The coordinator has to amend it.
- **ADV-MLM-1, whether anyone reaches it:** I packed the real release at seq_len 64 and 32. The train file differed both times (64 gave 2,099,407 sequences, which is 2 × 1,049,703 + 1). The held-out file was byte-identical at 32, 64 and 128. So the collision does not happen on the real release at those lengths; it does on a corpus I built for it. Rated INFEASIBLE.

**5. Attacked and could not break:** the tag 3 and tag 5 counter layouts against § 2.3; little-endian ids; `[SEP]` count against document count (fixture and real release); `--out` refusals; byte-identical reruns; `ess specify validate` (still 2 warnings; one new `unsynthesizable` entry for the Binary64 baseline, the same kind as the 16 already there); `task docs-check` exits 0; `cli.md` lists `data pack`.

**6. Paths written outside the worktree** (all under `<wave-root>/mlm/adv1-scratch/`):
- `mutants.sh`, `mutants.log`, `suite.log`, `validate-base.json`, `validate-head.json`
- Already removed: `mut/` (its target cleaned with `cargo clean --target-dir`, 2.1 GiB), `real-64/`, `real-32/`, `base/`.
- Also written: build output in the worktree's own `target/`, including `target/tmp/adversary_mlm_*`.

The lease is ended.

```findings
- file: crates/modelspec-runtime/src/dataset/mlm.rs
  line: 502
  category: correctness
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: packed_id hashes only the two packed files, so seq_len 2 and 4 on a corpus whose streams both divide give one packed_id for records differing in seq_len, sequence_count and binding_id; not reached on the real release at 32/64/128.
- file: crates/modelspec-runtime/tests/mlm.rs
  line: 218
  category: test-quality
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: derivation checks only element 0 and one-sequence HeldOut batches, so mutants using the in-sequence position as the counter index (identical masks across a batch) or w1 for the random token survive the unit suite.
- file: crates/modelspec-runtime/tests/mlm.rs
  line: 389
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the altered-file case changes the file length, so the blake3 check at mlm.rs:617 is never exercised and removing it survives the unit suite.
- file: crates/modelspec-runtime/tests/mlm.rs
  line: 316
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: no case reaches an epoch at or above 2^32, so dropping the epoch's high counter word survives the unit suite.
- file: crates/modelspec-runtime/schema/entities/modelspec.data.PackedDataset.schema.json
  line: 23
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the entity invariants seq_len > 0, vocabulary_size > 5 and baseline > 0 of PackedDataset and TrainingDataBinding are not projected, so records breaking them validate, the same class as ADV-CORPUS-2.
- file: crates/modelspec-runtime/src/dataset/mlm.rs
  line: 745
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: load accepts a hand-edited packed.json with vocabulary_size 4 and mlm_batch then panics on vocabulary_size - 5; only a hand-written record reaches it.
- file: .engineering/planning/story/bert-dropout.md
  line: 79
  category: docs
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the dropout story still assigns Philox stream tag 3, which D26 and DESIGN.md section 2.3 now give to masking with tag 4 reserved for dropout.
```
