---
format: aep.planning-md/3
id: story:content-hash-schema-constrained
kind: story
status: draft
title: The content-hash type and the distinct-roles rule are checked by the schema
relations:
- serves: vision:modelspec
- depends_on: story:bert-corpus-splits
revision: 1
---
# The content-hash type and the distinct-roles rule are checked by the schema

## Context

review-result:adversary-bert-corpus-pass-1, finding ADV-CORPUS-2 (pre-existing): the generated
schema of `modelspec.data.Dataset` projects `modelspec.artifact.ContentHash` as
`"type": "string"` only, and the `distinct: {in: splits, as: split, by: split.role}` invariant is
not projected, so a record with a `content_hash` that is not 64 lowercase hex digits, or with two
splits of one role, validates against the schema. The writer keeps both rules; the schema does
not, so "validates against the schema" is weaker than the record's contract. The same newtype
backs `artifact_id` digests and checkpoint records.

## Outcome

`modelspec.artifact.ContentHash` declares its alphabet and length in the specification (ESS
newtype `alphabet:` and a length invariant, `ess specify formats` lists what the pinned release
admits), the projection carries them (`pattern`, `minLength`, `maxLength`), and the two ignored
cases of `crates/modelspec-cli/tests/adversary_corpus_schema.rs` pass un-ignored. Where ESS
cannot project the distinct-roles invariant into JSON Schema, the story records that as an
`ess-limit` finding with the refusal or the missing projection quoted, files it on beyond10x/ess,
and keeps that case ignored with the issue link in the reason.

## Acceptance

- **hash-constrained**: `ess specify validate --path .` exits 0 with the constrained newtype; the
  generated `modelspec.data.Dataset.schema.json` refuses `"not a digest"` and accepts a 64-char
  lowercase hex string; `adv_cs_schema_refuses_a_content_hash_that_is_not_hex` passes un-ignored.
- **distinct-projected-or-filed**: `adv_cs_schema_refuses_two_splits_of_one_role` passes
  un-ignored, or the story's closing note quotes why ESS cannot project it and links the issue.
- **projections-current**: both schema projections regenerated; `task schema-check` exits 0;
  every reader of `ContentHash` (artifact, checkpoint, dataset) still reads its fixtures.
- `task ci` exits 0.

## Scope

- `spec/domains/artifact.yaml` (`ContentHash`), `crates/modelspec-runtime/schema/`,
  `crates/modelspec-observatory/` (generated)
- `crates/modelspec-cli/tests/adversary_corpus_schema.rs` (un-ignore)
