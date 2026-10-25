---
format: aep.planning-md/3
id: review-result:adversary-content-hash-pass-1
kind: review-result
status: active
title: Adversary pass 1 on content-hash-schema-constrained
relations:
- reviews: story:content-hash-schema-constrained
revision: 1
---
```
unit: story:content-hash-schema-constrained, worktree wt-6fcc5e03ccb1 at f21c94d plus 2 untracked adversary test files
verdict: CONFIRMED (no blockers: 1 issue-text note, 1 test-quality note, 1 pre-existing boundary note)
cases: executed 1388→1394, red 1
origin: introduced 2 / pre-existing 1 / undecided 0
wrote-outside-worktree: 1 path (the scratch directory below)
needs-coordinator: beyond10x/ess issue 686 needs one sentence corrected (finding 1); the bot or the coordinator does that, `gh` here is read-only
```

**1. What I touched.** `git --no-pager diff --stat` is empty because both new files are untracked. `git status --short` shows only my two test files, nothing else:
```
?? crates/modelspec-cli/tests/adversary_chash_wire.rs
?? crates/modelspec-cli/tests/adversary_chash_writers.rs
```

**2. Cases added** (all written before any suite run; the first run was `cargo test -p modelspec-cli --test adversary_chash_writers --test adversary_chash_wire`)

| Test | What it asserts | Now |
|---|---|---|
| `adversary_chash_writers.rs::adv_chash_compiled_artifact_hashes_are_content_hashes` | every hash field in `artifact.json` and each `kernels/*/cache_key/ir_hash` (cpu and vulkan compiles) passes the ContentHash schema and is 64 digits | green |
| `…::adv_chash_checkpoint_manifest_hashes_are_content_hashes` | same check for the `.msckpt` manifest's `schema_hash` and `payload_hash` | green |
| `…::adv_chash_data_records_hashes_are_content_hashes` | same check for both split hashes in `dataset.json` and `packed.json` | green |
| `…::adv_chash_the_check_refuses_each_malformed_form` | the check itself refuses uppercase, non-hex, 63, 65 and 0 digits | green |
| `adversary_chash_wire.rs::adv_chash_newtypes_are_bare_integers_and_a_string_is_refused` | `seq_len` and `vocabulary_size` are written as bare integers; a string, -1, 1.5, null or `{"value":16}` is refused by both schemas, and a string is refused by `mlm::load` | green |
| `…::adv_chash_schema_minimums_sit_at_the_writer_and_reader_bounds` | `pack --seq-len 1` writes records both schemas accept and `load` reads; `seq_len` 0 and `vocabulary_size` 5 are refused by schema and `load`; 6 is accepted | green |
| `…::adv_chash_schema_refuses_what_the_loader_refuses_at_the_upper_edge` | the schema refuses what `load` refuses at the upper edge | red, `#[ignore = "ADV-CHASH-1: …"]` |

Red output, run on its own with `--ignored`:
```
panicked at crates/modelspec-cli/tests/adversary_chash_wire.rs:227:5:
the schema accepts what load refuses: ["vocabulary_size 2147483649", "seq_len 16.0"]
test result: FAILED. 0 passed; 1 failed; 0 ignored
```

**3. Suite run** (after the cases existed): `b10x-build-slot cargo test --workspace` gave 1394 passed, 0 failed, 117 ignored, `EXIT=0`. My two files account for 6 passed and 1 ignored, which gives the 1388 "before" figure. `task schema-check` reports 100 artifacts current, `task docs-check` is current, and fmt and clippy (`-D warnings`) on modelspec-cli are clean.

**4. Findings**
- **F1 (issue 686 text):** issue 686 says the `distinct:` invariants on `CompiledArtifact.parameters` (by path and by canonical id), `CompiledArtifact.data` and `ModelGraph.nodes` reach the schema "by annotation only". They are entity invariants, and the entity schemas carry no top-level `x-ess-invariants`. The only `distinct` string in `CompiledArtifact.schema.json` is PlanStep's, at line 387, and `ModelGraph.schema.json` has none. The issue's companion 688 says the same thing correctly. Nothing reads the issue text programmatically.
- **F2 (test quality):** setting `VocabularySize` to `value > 6` (so `minimum` becomes 7) leaves every one of the unit's own tests green. Only my new bounds case fails, at `adversary_chash_wire.rs:184`.
- **F3 (ADV-CHASH-1, pre-existing):** the PackedDataset schema accepts `vocabulary_size` 2^31+1 and `seq_len` 16.0, while `mlm::load` refuses both (`mlm.rs:600` and serde's u64). At the base commit both fields were a bare `"type":"integer"`. No writer produces either value, so this is INFEASIBLE.

**5. Attacked and could not break**
- **Attack 1:** every reader that verifies a hash (`load`, `pack`, the artifact reader, the checkpoint reader) recomputes it and compares strings. So "", 63 or 65 digits, and uppercase are all refused; none compares strings without recomputing.
- **Attack 2:** no test fixture or literal holds a hash that isn't 64 lowercase hex digits.
- **Attack 3:** the UI types (`observatory.ts`) contain no data records. No source file changed, so the writers produce the same bytes as before.
- **Attack 4:** `pack --seq-len 0` exits 2 (`data_pack.rs:469`). `load` refuses `seq_len` 0, but its message says "positive sequence length" rather than naming `seq_len`; that message predates this unit. The generated docs never render the data domain.
- **Attack 5:** issues 685, 687 and 688 match the tree. I re-checked the length forms 685 claims also fail to project (`>=`/`<=`, `{gte,lte}`, `utf8_bytes`): all three still produce no minLength/maxLength.
- **Attack 6:** base and head have identical completeness: unanswered 0, unsynthesizable 17, 2 warnings.
- **Attack 7:** dropping the alphabet, `value >= 0` on SequenceLength, swapping the two invariants, adding uppercase to the alphabet, removing `f` from the alphabet, and `value > 1` on SequenceLength are each caught by the unit's own tests. F2 is the one survivor.

**6. Written outside the worktree:** `<wave-root>/content-hash/adv1-scratch` (16M). It holds the base and head spec copies, the probes, the mutant copy with its target already removed by `cargo clean`, `run-mutant.sh` and the logs.

```findings
- file: https://github.com/beyond10x/ess/issues/686
  category: docs
  severity: note
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The issue says the distinct invariants of CompiledArtifact.parameters, CompiledArtifact.data and ModelGraph.nodes reach the schema as annotations, but as entity invariants they reach it not at all (issue 688).
- file: crates/modelspec-cli/tests/adversary_chash_wire.rs
  line: 184
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: A VocabularySize mutant to value > 6 (minimum 7) survives every test the unit wrote and is caught only by the new adversary bounds case.
- file: crates/modelspec-runtime/schema/entities/modelspec.data.PackedDataset.schema.json
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: pre-existing
  message: The schema accepts vocabulary_size 2^31+1 and seq_len 16.0, which mlm::load refuses, but no writer produces either (ADV-CHASH-1).
```
