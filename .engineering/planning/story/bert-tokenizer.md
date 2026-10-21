---
format: aep.planning-md/3
id: story:bert-tokenizer
kind: story
status: draft
title: Train a tokenizer on the train split in Rust, with an exact vocabulary size
summary: Adds modelspec data tokenizer, trained on the Train split only to an exact vocabulary size, reproducibly; settles the TokenizerKind and Vocabulary special-token markers.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-corpus-splits
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: crates/modelspec-cli/src/data.rs
- confidence: cited
  path: crates/modelspec-cli/src/lib.rs
- confidence: cited
  path: crates/modelspec-cli/tests/data_tokenizer.rs
- confidence: cited
  path: crates/modelspec-runtime/schema/
- confidence: cited
  path: crates/modelspec-runtime/src/dataset/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/src/dataset/tokenizer.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/tokenizer.rs
- confidence: cited
  path: spec/domains/data.yaml
- confidence: cited
  path: website/docs/reference/cli.md
revision: 3
---
# Train a tokenizer on the train split in Rust, with an exact vocabulary size

## Context

`epic:bert-from-scratch` (In scope, Data: "a WordPiece (or byte-level) tokenizer"; Model size: "vocabulary
about 30k"). `spec/domains/data.yaml` types `modelspec.data.Tokenizer` with a `kind`
(`modelspec.data.TokenizerKind`, `WordPiece` or `ByteLevel`) and a `vocabulary`
(`modelspec.data.Vocabulary`: `tokens`, a token id being its position, and `mask_token_id`).

## Open questions settled first

Two `UNMAPPED:` markers of `spec/domains/data.yaml` stand in front of this story, named by where
they sit and a quoted fragment of their text, never by line number. Each is settled before any
code, in that file, with `ess specify validate --path .` in `spec/` passing afterwards. A
settlement names its source (the operator, DESIGN.md or another ESS document); where none answers,
the story stops at that step and reports the question instead of choosing.

1. Marker on `modelspec.data.TokenizerKind`, text "the epic leaves WordPiece or byte-level open":
   which of the two; the other variant is removed.
2. Marker on `modelspec.data.Vocabulary`, text "special tokens beyond the mask token": which
   special tokens there are (classification, separator, padding, unknown), their ids, and a field
   per token the pipeline must find, with invariants like those of `mask_token_id`.

## What this story decides

- `modelspec data tokenizer --dataset <dir> --vocab-size <n> --out <file>` trains the settled
  algorithm on the `Train` split of a dataset written by `story:bert-corpus-splits`, never on the
  `HeldOut` split, and writes a JSON `Tokenizer` record. The vocabulary has exactly `n` tokens,
  specials included; a corpus too small to yield `n` tokens exits 1 naming the count it reached.
- Training is deterministic: ties between equally frequent candidates break by byte order, so two
  runs write byte-identical files, and the tokenizer id is derived from content as in
  `story:bert-corpus-splits`.
- The algorithm is written in this repository (no tokenizer crate), in Rust, so that its tie rules
  and its output are fixed by the code here. `encode` and `decode` are library functions the later
  stories call.
- I/O and refusals behave as `modelspec data prepare` does (exit 1 with a message, exit 2 on usage
  errors, no diagnostic code).

## Order on shared files

This story edits `spec/domains/data.yaml`, `crates/modelspec-runtime/src/dataset/`,
`crates/modelspec-runtime/schema/`, `crates/modelspec-cli/src/data.rs` and `src/lib.rs`, and
`website/docs/reference/cli.md`, which `story:bert-corpus-splits` creates or edits first, so it
depends on that story. `story:bert-mlm-batches` follows it. It edits no file of the operator
stories.

## Domain relations

- `modelspec.data.Tokenizer` holds its vocabulary as a field of type `modelspec.data.Vocabulary`
  (`spec/domains/data.yaml`): part of the tokenizer value, with no identity of its own.
- `Vocabulary.mask_token_id` names a position in `Vocabulary.tokens` through the invariants
  `mask_token_id >= 0` and `mask_token_id < tokens.count` (`spec/domains/data.yaml`); the settled
  special tokens follow the same form.
- Which dataset a tokenizer was trained on is not declared in `spec/domains/data.yaml`; this story
  declares no relation for it. Where a run records the tokenizer it used is the marker at the top
  of `spec/domains/data.yaml`, settled by `story:bert-mlm-batches`.

## Scope

- `spec/domains/data.yaml` (the two settlements); `crates/modelspec-runtime/schema/` (regenerated).
- `crates/modelspec-runtime/src/dataset/tokenizer.rs` (new), `dataset/mod.rs`.
- `crates/modelspec-cli/src/data.rs`, `src/lib.rs` (the `tokenizer` verb).
- `crates/modelspec-runtime/tests/tokenizer.rs` (new), `crates/modelspec-cli/tests/data_tokenizer.rs`
  (new); fixtures from `story:bert-corpus-splits`.
- `AGENTS.md` (Commands); `website/docs/reference/` through `task docs-generate`. No
  `CHANGELOG.md` line (`story:bert-train-data-binding` covers the data commands).

## Acceptance

`modelspec data tokenizer` trains the settled tokenizer on the fixture `Train` split to exactly the
requested vocabulary size, reproducibly, with the settled special tokens, and `decode(encode(text))`
gives back the text as the settled normalization defines it.

Scenarios, each run by `task ci` through the command shown, except scenario 1:

1. `spec/domains/data.yaml` no longer contains the markers on `TokenizerKind` ("the epic leaves
   WordPiece or byte-level open") and on `Vocabulary` ("special tokens beyond the mask token");
   typed declarations that cite their source stand in their place, and
   `cd spec && ess specify validate --path .` exits 0.
2. `cargo test -p modelspec-cli --test data_tokenizer exact_size`: with `--vocab-size 300` on the
   fixture dataset, the record has exactly 300 tokens, no duplicates, every settled special token at
   its settled id, and it validates against the generated schema; two runs write identical bytes.
3. `cargo test -p modelspec-runtime --test tokenizer round_trip`: every fixture document, both
   splits, survives `decode(encode(text))` as the settlement defines; no encoded id is outside
   `[0, 300)`.
4. `cargo test -p modelspec-runtime --test tokenizer train_split_only`: a word that the fixture puts
   only in the `HeldOut` split is not a single vocabulary token, while a word of equal frequency in
   the `Train` split is.
5. `cargo test -p modelspec-cli --test data_tokenizer too_small`: `--vocab-size 100000` on the
   fixture exits 1 naming the size reached.
6. `task docs-check` exits 0, and `website/docs/reference/cli.md` lists `data tokenizer`.

## Out of scope

- The vocabulary size of the real run (`story:bert-tiny-cpu-model` fixes the model at 30522;
  `story:bert-tiny-gpu-pretraining` trains the tokenizer at that size).
- Packing and masking (`story:bert-mlm-batches`).
