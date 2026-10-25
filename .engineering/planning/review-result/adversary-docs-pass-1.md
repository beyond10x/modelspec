---
format: aep.planning-md/3
id: review-result:adversary-docs-pass-1
kind: review-result
status: active
title: Adversary pass 1 on story:docs-current-bert-1
relations:
- reviews: story:docs-current-bert-1
revision: 1
---
unit: story:docs-current-bert-1, worktree wt-41996e980b07 at HEAD ce3032b (base 08639c3), plus my one untracked test file
verdict: findings 7 (blocker 0, warning 2, note 5); 4 of them are red cases and all are CONFIRMED
cases: executed 60→66, red 4 (the 4 red cases are `#[ignore]`d and fail under `--ignored`)
origin: introduced 5 / pre-existing 2 / undecided 0
wrote-outside-worktree: 1 path (adv1-scratch, listed in part 6)
needs-coordinator: whether the missing `lede`/`source`/`sidebar_position` (front matter the `docs` skill requires but every older hand-written page also lacks) holds this unit or becomes a story; I could not verify the pitch deck's "2440 passed" myself because the brief forbids a full-workspace test run

**1. Diff of the tree**

`git --no-pager diff --stat` is empty because my file is untracked. `git status --short`:
```
?? crates/modelspec-docs/tests/adversary_docs.rs
```
It is a test file. I touched no other path.

**2. Cases added** (`crates/modelspec-docs/tests/adversary_docs.rs`)

| case | asserts | now |
|---|---|---|
| `adversary_docs_1_new_concept_and_guide_pages_carry_the_skill_front_matter` | the two new pages carry `sidebar_position`, `lede`, `source` | red |
| `adversary_docs_2_the_overview_mnist_claim_agrees_with_the_benchmark_history` | the `index.md` claim matches `benchmarks/runs.yaml` | red |
| `adversary_docs_3_the_mlm_batch_row_names_its_release` | the MLM batch row starts "Since 0.10.0" | red |
| `adversary_docs_4_the_pitch_notes_cite_changelog_lines_of_the_release_they_name` | each `CHANGELOG.md:N` cite in the deck falls in the release it names | red |
| `adversary_the_data_guide_transcripts_equal_a_run_in_the_tree` | all 4 guide commands, run in-process, give the pasted bytes, including the binding table | green |
| `adversary_the_getting_started_transcript_equals_a_run_in_the_tree` | acceptance 5 | green |
| `adversary_the_bert_page_e_target_011_output_equals_a_run` | `ir --stage kernel` and `compile --backend vulkan` print the pasted E-TARGET-011 and exit 1 | green |
| `adversary_the_pipeline_page_ir_listings_equal_a_run` | the tensor and training IR on the pipeline page | green |
| `adversary_docs_check_refuses_a_shipped_row_whose_test_is_gone` | mutant (a), in a sandbox copy | green |
| `adversary_docs_check_refuses_a_raw_admonition_on_a_new_page` | mutant (d), in a sandbox copy | green |

Red output from running each case alone, before the suite:
```
front matter missing: ["concepts/bert-operators.md: sidebar_position", "concepts/bert-operators.md: lede", "concepts/bert-operators.md: source", "guides/prepare-training-data.md: sidebar_position", "guides/prepare-training-data.md: lede", "guides/prepare-training-data.md: source"]
index.md says "no new timed step time was measured for any later release", and runs.yaml times the MNIST shapes later: ["sweep-b100-h64-modelspec-vulkan-0.9.0 (33.04433734408971 µs)", "sweep-b100-h64-modelspec-vulkan-0.10.0 (30.725802505377537 µs)"]
the row does not name its release: 15% of the tokens are selected from the seed, and of those 80% are masked, 10% replaced by a random token and 10% kept. Held by test proportions in crates/modelspec-runtime/tests/mlm.rs.
"CHANGELOG.md:81 cited as 0.4.0, is in 0.7.1", … 14 entries, … "CHANGELOG.md:131 cited as 0.1.0, is in 0.4.0"
```

**3. Suite run** (`b10x-build-slot cargo test -p modelspec-docs`, exit 0)
```
unittests src/main.rs        test result: ok. 51 passed; 0 failed; 0 ignored
tests/adversary_docs.rs      test result: ok. 6 passed; 0 failed; 4 ignored
tests/benchmarks_adversary.rs test result: ok. 9 passed; 0 failed; 0 ignored
--ignored on adversary_docs: test result: FAILED. 0 passed; 4 failed
```
The 60 in "executed 60→66" comes from this same run with my file left out (51 + 9).

**4. Findings**

| id | file:line | severity | verdict | origin | what was measured / what reaches it |
|---|---|---|---|---|---|
| ADV-DOCS-1 | `website/docs/concepts/bert-operators.md:1`, `guides/prepare-training-data.md:1` | warning | CONFIRMED | introduced | Red case 1. The `docs` skill (lines 87–88) requires these fields. Every older hand-written page lacks them too, so that part is pre-existing. |
| ADV-DOCS-2 | `website/docs/index.md:71-72` | warning | CONFIRMED | introduced | Red case 2. The unit widened "0.7.0 or 0.7.1" to "any later release". The 0.9.0 and 0.10.0 sweeps time the same 784→64→10, batch-100 shapes, and the Benchmarks page the same paragraph links charts them. They ran through the sweep harness, not `train`, so the sentence is misleading rather than false. |
| ADV-DOCS-3 | `crates/modelspec-docs/src/status.rs` ("Masked-language-model batches" row) | note | CONFIRMED | introduced | Red case 3. `status.mdx:265-266` says the prefix names the release. `tests/mlm.rs::proportions` is in the 0.10.0 tag. |
| ADV-DOCS-4 | `docs/pitch/index.html:547,574,645,704` | note | CONFIRMED | pre-existing | Red case 4. The cites were already stale at 08639c3 (0.1.0 was at line 187, cited as 131). The unit rewrote these notes but kept the stale cites. |
| ADV-DOCS-5 | `docs/pitch/index.html:547,574,645,669,704` | note | CONFIRMED | pre-existing | The publicly served speaker notes still hold 9 `story:`/`epic:` ids and worklog text. The unit removed 5 ids and added none. |
| ADV-DOCS-6 | `crates/modelspec-docs/src/status.rs` (`the_list_was_checked_against_the_0_10_0_release`) | note | CONFIRMED | introduced | This test only pins a constant (`AS_OF >= "2026-10-25"`) and observes no behaviour. Likewise the `<= USAGE_WIDTH` bound in `cli.rs:230` moves with the constant it checks; only the literal `data pack` line catches the width mutant. |
| ADV-DOCS-7 | `crates/modelspec-core/src/compile.rs:42`, `website/docusaurus.config.ts:67` | note | CONFIRMED | introduced | Both files are outside the story's scope list: compile.rs is a doc comment only, the config file is the edit URL fix. `website/build` (12:10) is older than the commit (12:18), so acceptance 2 was read from a build before those two patches. |

Mutants:

| mutant | killed by |
|---|---|
| (a) shipped row's test renamed | `generate --check`: "defines no fn sequences" (my green case) |
| (b) `FONT` 15→11, scratch copy | `chart_text_is_at_least_12_px_in_a_390_px_viewport`, `the_committed_page_and_charts_equal_the_generated_ones` |
| (c) `USAGE_WIDTH` 64→80, scratch copy | only the literal in `usage_lines_fit…` (`cli.rs:230`), plus `every_command_option_and_possible_value_is_listed` |
| (d) `:::caution Planned` | `generate --check`: raw admonition check (my green case) |

**5. Attacked and not broken**
- Every guide transcript: ids and blake3 digests match byte for byte.
- The getting-started transcript, the pipeline IR and the `--stage kernel` table.
- E-TARGET-011 output and exit 1 on `ir`, `compile`, `train` and `infer --backend vulkan`; `--eval` beside a packed `--data` exits 2.
- 17 operators in `resolve.rs:64`; Philox tags 3/4/5; E-TYPE-001/002/011/014/018/019/022 probed with `check`; E-DATA-001…009 present.
- 50 shipped rows: every evidence is a `#[test]`; only Vulkan or browser tests are `#[ignore]`.
- 5 planned rows are draft in the store; no story id on any page; the data commands are in the 0.10.0 tree with no CHANGELOG entry.
- Every relative link and anchor resolves; no `{` or capitalised tag outside code.
- Charts: no white fill, a dark block, 15 × 358/440 = 12.2 px. With the OS scheme and site theme set opposite, Chromium follows the site theme (checked with Playwright).
- The 20 template file:line cites hold their rules; s1-07 names only a 12-line wrapper, as the implementor said.
- `xmllint` passes the deck; 50/5 rows and 12 tags match.
- `docs-check` exits 0; no test reads AGENTS.md; AGENTS.md's `path::function` references exist.

**6. Paths written outside the worktree**
- `<wave-root>/current/adv1-scratch/` (21M): `run1/`, `gs/`, `probes/`, `shots/`, `mutant/` (its target removed with `b10x-build-slot cargo clean`), `benchmarks.rs.orig`, `cli.rs.orig`, `ids-base.txt`, `ids-head.txt`, `serve.log`. The Node probe was deleted.
- Inside the tree's gitignored `target/`: `target/tmp/adversary-docs/` (40M of sandbox copies) and the build products.

**7. Findings block**
```findings
- file: website/docs/concepts/bert-operators.md
  line: 1
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the new concept and guide pages lack the sidebar_position, lede and source front matter the docs skill requires
- file: website/docs/index.md
  line: 71
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the overview says the MNIST shapes were not timed in any release after 0.6.0, but runs.yaml times them in 0.9.0 and 0.10.0
- file: crates/modelspec-docs/src/status.rs
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the Masked-language-model batches row shipped in 0.10.0 but has no "Since 0.10.0", although the status page says that prefix names the release
- file: docs/pitch/index.html
  line: 704
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: 14 CHANGELOG.md line cites in the pitch speaker notes point into other releases' sections
- file: docs/pitch/index.html
  line: 547
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: the publicly served pitch speaker notes still hold 9 planning ids and internal worklog text
- file: crates/modelspec-docs/src/status.rs
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the_list_was_checked_against_the_0_10_0_release only pins a constant, and the cli usage-width bound moves with the constant it checks
- file: crates/modelspec-core/src/compile.rs
  line: 42
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: compile.rs and docusaurus.config.ts changed outside the story scope, and the website build behind acceptance 2 predates those patches
```
