---
format: aep.planning-md/3
id: review-result:adversary-bert-tokenizer-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-tokenizer
relations:
- reviews: story:bert-tokenizer
revision: 1
---
```
unit: story:bert-tokenizer (worktree wt-a9a168f6af78, branch impl/bert-tokenizer, head 0626dda, plus my two untracked test files)
verdict: red
cases: executed 1271→1288, red 2
origin: introduced
wrote-outside-worktree: <wave-root>/tokenizer/adv1-scratch/ (mut/ with its own 1.9G target/, mutants.sh, mutants.log, mutants.out, probe_real.rs, real-tokenizer.json, real-M9.json, real-M11-frequency.json, real-M13.json, real-time.txt, suite.log)
needs-coordinator: D23's merge score makes the real 8,192 vocabulary character-level for English (ADV-TOK-2); the decision itself has to change, not the code
```

**1. What I touched.** `git --no-pager diff --stat` is empty: no tracked file changed. `git status --short` shows only my two test files:
```
?? crates/modelspec-cli/tests/adversary_tokenizer_cli.rs
?? crates/modelspec-runtime/tests/adversary_tokenizer_1.rs
```
No path outside `crates/*/tests/`. I wrote no fixture file. No `/home/` path appears in either file.

**2. Cases I added.** Each was run alone before the suite.

The two red cases are `#[ignore]`d with their finding id. This is their first run (`cargo test -p modelspec-runtime --test adversary_tokenizer_1 -- --ignored`):
```
adv_score_cross_product_past_u128: panicked at crates/modelspec-runtime/src/dataset/tokenizer.rs:281:20: attempt to multiply with overflow
adv_every_size_up_to_the_maximum_trains: 4 sizes refused: ["6: the split yields 5 tokens, fewer than the 6 requested", "8: the split yields 7 tokens, fewer than the 8 requested", "10: the split yields 9 tokens, fewer than the 10 requested", "12: the split yields 11 tokens, fewer than the 12 requested"]
```

The green cases now pass. `adversary_tokenizer_1.rs` has 10 passed and 2 ignored; `adversary_tokenizer_cli.rs` has 7 passed. In the runtime file:
- `adv_encode_pins_longest_match_segmentation`, `adv_tie_breaks_by_merged_token_not_first_part` and `adv_alphabet_ties_break_by_byte_order` pin encoding and tie order to literal tokens.
- `adv_score_at_real_scale_counts_is_exact` uses counts of 3e9, where the cross-products pass `u64`.
- The rest cover normalization edge cases, special-token text, the smallest sizes, the fixture maximum (314), the 100-character cap and the record id.

The CLI file covers sizes 0, 4, -1, `1e3` and 2^64, `--out` that is a directory, `--dataset` that is a file, a missing `heldout.raw`, invalid UTF-8, `--out` inside the dataset or already existing, and schema `const`s.

Mutants, run in the scratch copy only (`mutants.log`):

| mutant | unit's suite | my cases |
|---|---|---|
| M1 encode shortest match first (:505) | **survives** | killed |
| M2 tie rule dropped | killed | killed |
| M3 alphabet ties by frequency only | killed only by chance (depends on hash order) | killed every run |
| M4 `u64` products (:276-282) | **survives** | killed |
| M5 no `##` on continuation tokens | killed | killed |
| M6 every word encodes to `[UNK]` | killed | killed |
| M7 id from the token count | killed | killed |
| M8 also trains on HeldOut | killed | killed |
| M9 skip the "already in vocabulary" check | survives | survives; same real output at 8,192 |
| M10 trainer word cap `>=` (:395) | **survives** | killed |
| M11 score by pair frequency | killed | killed |
| M12 encode cap `>=` | killed | survives |
| M13 tie by first part instead of merged token (:294) | **survives** | killed; real vocabulary changes (974 diff lines) |

**3. Suite run** (after the cases existed): `b10x-build-slot cargo test --workspace` gave `EXIT=0`, 1288 passed, 0 failed, 111 ignored. My two files account for 7 + 10 of those. The before count of 1271 is that same run minus my two test binaries; I did not run the suite beforehand.

**4. Findings** (they cover 0626dda):
- **ADV-TOK-2 (blocker)**, `tokenizer.rs:276` and the `Tokenizer` summary in `data.yaml`. Measured on the real release at 8,192 (`probe_real.rs`):
  - 0 of the 6,187 merged tokens consist only of ASCII letters or digits.
  - The first merges are rare CJK and Khmer pairs; `the` encodes as `t ##h ##e`.
  - HeldOut: 4.82 ids per alphanumeric word, and 4.28% of words are a single token.
  - The same trainer ranking pairs by frequency instead (M11) gives 1.32 ids per word, 79.3% single-token words and 280,149 HeldOut ids against 923,211.
  - What reaches it: the real run (`--vocab-size 8192`, D19).
  - The spec says the score rule is "as the published trainer of the `tokenizers` library". I believe that trainer merges by frequency (its WordPiece trainer wraps the BPE trainer), but that is from memory; I did not check the library source here.
- **ADV-TOK-1 (note)**, `tokenizer.rs:208`. Sizes 6, 8, 10 and 12 are refused with "the split yields N tokens", although the same split yields 314. The cause is the capped alphabet: these sizes need a merge but have too few characters to make one. What reaches it: the `--vocab-size` flag; no documented run asks for under 2,005.
- **ADV-TOK-3 (warning), 4 entries.** Mutants M1, M4, M10 and M13 survive the unit's suite. M4 would wrap silently in a release build at real counts, and M13 changes the real vocabulary. My cases now in the tree kill all four.
- **ADV-TOK-4 (note, INFEASIBLE)**, `tokenizer.rs:281`. The comparison multiplies three `u64` values, so it overflows `u128` at counts of 2^43. That needs about 8.8e12 words, so it was built, not reached.
- **ADV-TOK-5 (note)**, `data.yaml`. The summary says normalization works "as BERT's BasicTokenizer does". Format characters (ZWJ, soft hyphen) become words of their own here, and the real Train split has 1,248 such words. My recollection is that BERT drops these; not checked against BERT's source.

**5. Attacked without breaking it:**
- Two separate processes write byte-identical output, and my real run equals the implementor's file.
- Real wall time 9.8 s (not 24.4 s; the machine was shared). I could not measure RSS because `/usr/bin/time` is missing.
- HeldOut `[UNK]` rate is 0.0235% (52 words), well under 1%.
- The alphabet cap leaves out 3,723 characters, which make up 0.003% of character occurrences.
- HeldOut is never opened, invalid UTF-8 exits 1, no `.partial` file is left behind, and the special ids are `const` in the schema.
- `task docs-check`, `ess specify validate` and `task schema-check` pass.
- The verb sits in AGENTS.md's State paragraph next to `data prepare`, so it is consistent there, though the story's scope said Commands.
- `distinct` is not in the generated schema (no `uniqueItems`), as it already was at base; `exact_size` checks distinctness itself.
- Training on the raw text instead of the word table is equivalent: same counts (argued, not run).
- M9's branch was reached by no input I could build and not by the real run.

**6. Paths written outside the worktree:** everything is under `<wave-root>/tokenizer/adv1-scratch/`, as in the header. Remove `mut/target` with `cargo clean --target-dir` once the unit merges. Lease ended.

```findings
- file: crates/modelspec-runtime/src/dataset/tokenizer.rs
  line: 276
  category: correctness
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The D23 score rule leaves the real 8,192 vocabulary with no ASCII-only merge at all, so English encodes at 4.82 ids per word against 1.32 when pairs are ranked by frequency, which defeats the stated aim of comparing with published vocabularies.
- file: crates/modelspec-runtime/src/dataset/tokenizer.rs
  line: 208
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: Sizes 6, 8, 10 and 12 are refused as "the split yields N tokens" on a split that trains to 314, because the capped alphabet leaves no pair to merge.
- file: crates/modelspec-runtime/src/dataset/tokenizer.rs
  line: 505
  category: test-quality
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: An encoder that takes the shortest match first survives the unit's suite, because round_trip only compares decoded text.
- file: crates/modelspec-runtime/src/dataset/tokenizer.rs
  line: 294
  category: test-quality
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: Breaking ties by the first part instead of the merged token survives the unit's suite and changes the real vocabulary.
- file: crates/modelspec-runtime/src/dataset/tokenizer.rs
  line: 281
  category: test-quality
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: Doing the score arithmetic in u64 survives the unit's suite, although real-release counts overflow u64.
- file: crates/modelspec-runtime/src/dataset/tokenizer.rs
  line: 395
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: Changing the trainer's 100-character cap from > to >= survives the unit's suite.
- file: crates/modelspec-runtime/src/dataset/tokenizer.rs
  line: 281
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: The score comparison multiplies three u64 values and overflows u128 at 2^43 counts, which contradicts "compared exactly" but no corpus here reaches it.
- file: spec/domains/data.yaml
  category: docs
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The summary says normalization works as BERT's BasicTokenizer does, yet format characters such as ZWJ and soft hyphen become words of their own.
```
