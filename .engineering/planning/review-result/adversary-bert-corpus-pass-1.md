---
format: aep.planning-md/3
id: review-result:adversary-bert-corpus-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-corpus-splits
relations:
- reviews: story:bert-corpus-splits
revision: 1
---
```
unit: story:bert-corpus-splits, commit 674c351 on impl/bert-corpus-splits (worktree wt-80e4bdccf4f3), plus my two untracked test files
verdict: red
cases: executed 1226→1238, red 5
origin: introduced
wrote-outside-worktree: <wave-root>/corpus/adv1-scratch (1 directory, 518M)
needs-coordinator: an article rule that gives 28,475 on the real train split. The blank-line rule gives 28,472, and I don't know where the other 3 are.
```

**1. Diff stat**

`git --no-pager diff --stat` shows only `.engineering/waves/bert-1.md | 3 ++-`. That edit was already there before I took the lease (my first `git status` showed it), so it is the coordinator's, not mine. I added two untracked files, both tests:
- `crates/modelspec-runtime/tests/adversary_corpus_1.rs`
- `crates/modelspec-cli/tests/adversary_corpus_schema.rs`

No implementation, spec, schema or docs file was changed.

**2. Cases added. Each red case was run alone first, with `--ignored`.**

| Case | Now | Red output when written |
|---|---|---|
| `adv_c1_table_key_row_is_not_an_article` | ignored, reason ADV-CORPUS-1 | `left: 2 right: 1` (adversary_corpus_1.rs:74) |
| `adv_c1_equation_line_is_not_an_article` | ignored, reason ADV-CORPUS-1 | `left: 2 right: 1` (:88) |
| `adv_c1_real_release_counts_match_the_published_article_counts` (needs `MODELSPEC_WIKITEXT103_RAW`, deletes its copy afterwards) | ignored, reason ADV-CORPUS-1 | `left: [29444, 60] right: [28475, 60]` (:102) |
| `adv_cs_schema_refuses_a_content_hash_that_is_not_hex` | ignored, reason ADV-CORPUS-2 | `assertion failed: !v.is_valid(&record(vec![split("Train", "not a digest"), …]))` |
| `adv_cs_schema_refuses_two_splits_of_one_role` | ignored, reason ADV-CORPUS-2 | `assertion failed: !v.is_valid(&record(vec![split("Train", HASH), split("Train", HASH)]))` |
| 12 other cases (heading shapes; paths; line endings and encodings; refusals; dataset id; distinct roles; out is a file; out equals the corpus directory; valid schema record) | pass | none |

**3. Suite run**

- Command: `b10x-build-slot cargo test --workspace`. Exit 0: 1,238 passed, 0 failed, 108 ignored.
- My two test files ran 12 cases (11 + 1), so the count before my files is 1,226. I got that number by subtracting from this same run, not from a separate run.
- `cargo fmt --all` and `clippy -D warnings` are clean on both crates.

**Per attack**

1. **Red: the document rule (ADV-CORPUS-1).**
   - I ran `modelspec data prepare` on the real release. Train gives 29,444 documents against the published 28,475 (+969). Valid gives 60, which matches.
   - A heading line is ` = Homarus gammarus = \n`: one leading space, one trailing space. A blank line is a single space followed by a newline.
   - The matcher at `corpus.rs:21` also accepts lines inside articles, for example table keys (` = Position ; GP = `) and equation fragments (` = 14 @.@ 1 years ) , 115mCd ( t1 / 2 = `). Section headings are correctly rejected.
   - Taking only heading lines with a blank line directly before and after gives 28,472 / 60 / 60 for train / valid / test. Probed with awk, not committed.
   - The spec's wording at `data.yaml:117` is literally what the code does. The spec says "one article", but the rule does not produce one article per document.
   - What reaches it: the only corpus the spec names, through the documented command.
   - Why the tests missed it: the oracle `article_titles` in `data_prepare.rs:64` is the same matcher written again, so the `splits` test cannot catch a wrong rule.
2. **Green: reproducibility across paths.** A trailing slash or a `dir/../dir/.` path writes identical bytes. A different directory name changes `name` but not `dataset_id`, and the spec says so. `data_prepare.rs` already runs the two prepares in two separate processes.
3. **Green: line endings and encodings.** CRLF, a UTF-8 BOM, no final newline and a one-line file all give the right counts. `byte_count` equals the file length, and the blake3 of the copy equals the blake3 of the source. An empty file, a file of only blank lines, and invalid UTF-8 (reported at byte 17) all exit 1 and leave no files behind.
4. **Green: the dataset id.** It is the leading 16 bytes of blake3 over the Train digest then the HeldOut digest, the same construction as `artifact_id` (`artifact.rs:631`). Changing only the valid split changes the id. The suite I was handed has no test of this (ADV-CORPUS-3).
5. **Red: the schema (ADV-CORPUS-2).** The record validation in the tests is real: `jsonschema` `iter_errors`. But `ContentHash` is only `"type": "string"` (`Dataset.schema.json:40`), and the "distinct by role" rule is not in the schema at all. The base commit's projection has the same gaps, so this is pre-existing. The writer does keep roles distinct even when both files are identical.
6. **Green: refusal paths.** A failing run leaves an earlier output untouched. `--out` pointing at a file is refused and the error names it. `--out` equal to the corpus directory leaves `wiki.*.raw` alone. One side effect: a run that fails after the checks still creates an empty `--out` directory.
7. **Green: the drift check.** Deleting `.ess-output/state.json` makes the check exit 1 ("missing published output-state checkpoint"). An extra file passes, as AGENTS.md documents. Nothing reads `crates/modelspec-runtime/schema/` at build time; only `Taskfile.yml` and the tests do.
8. **Green: spec settlements.** `ess specify validate --format json` exits 0 with the same 18 completeness entries as the base. `mask_token_id == 4` is projected as `"const": 4`, so the schema checks it. Philox tag 3 is prose only, so a later story has to know it; the summary says story:bert-mlm-batches adds it to DESIGN.md § 2.3.
9. **Green: docs.** `modelspec-docs generate --check` exits 0. `cli.md:76` lists `data prepare`, and exit codes are in the global § Exit status. AGENTS.md is consistent with the code.
10. **Test quality (ADV-CORPUS-3).**
    - Mutants the existing tests catch, by inspection: deleting the heading check (`corpus.rs:133`, `data_prepare.rs:111`) and a random id (`reproducible`).
    - Mutants that survive, also by inspection and not built: swapping the hash order, hashing the hex text, or hashing Train only. The only id assertion is `data_prepare.rs:134`, which checks the version nibble.
    - Deleting the cleanup on error (`mod.rs:341-345`) also survives: every failure test stops before `create_dir_all`.
    - My `adv_c1_dataset_id_is_blake3_of_train_then_heldout_digests` and `adv_c1_refused_run_keeps_an_earlier_output_intact` catch these mutants.

**Written outside the worktree:** everything is under `<wave-root>/corpus/adv1-scratch/`:
- `real/`: the real-release output, about 540 MB of corpus copies.
- `base/`: the base spec.
- `drift-extra/`, `drift-nostate/` and their `.log` files.
- `validate*.json`, `validate.err`, `suite.log`.

Inside the worktree, under `target/`: the release CLI build, and my own real-release copy in `target/tmp`, which I deleted after measuring.

```findings
- file: crates/modelspec-runtime/src/dataset/corpus.rs
  line: 21
  category: correctness
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The heading matcher also counts table-key and equation lines inside articles, so the real train split records 29,444 documents against the published 28,475 article count (blank-delimited headings give 28,472), and the spec rule at spec/domains/data.yaml:117 states the same wrong rule.
- file: crates/modelspec-cli/tests/data_prepare.rs
  line: 64
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The document-count oracle repeats the implementation's matcher, so the splits scenario cannot detect a wrong article rule.
- file: crates/modelspec-runtime/schema/entities/modelspec.data.Dataset.schema.json
  line: 40
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: The generated schema accepts a content_hash that is not 64 lowercase hex digits and a record with two splits of one role, so "validates against the schema" is weaker than the record's contract.
- file: crates/modelspec-cli/tests/data_prepare.rs
  line: 134
  category: test-quality
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: No test of the unit pins dataset_id to its stated derivation or exercises prepare's partial-file cleanup at mod.rs:341, so swapping the role order or dropping the cleanup survives the suite.
```
