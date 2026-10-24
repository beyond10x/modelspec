---
format: aep.planning-md/3
id: story:bert-mlm-batches
kind: story
status: active
title: Pack token sequences and draw 80/10/10 MLM batches from Philox
summary: Adds modelspec data pack and a pure mlm_batch function with seeded data order and the held-out unigram baseline; settles the MaskingPolicy and run-record markers.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-tokenizer
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/data.rs
- confidence: cited
  path: crates/modelspec-cli/src/lib.rs
- confidence: cited
  path: crates/modelspec-cli/tests/data_pack.rs
- confidence: cited
  path: crates/modelspec-runtime/schema/
- confidence: cited
  path: crates/modelspec-runtime/src/dataset/mlm.rs
- confidence: cited
  path: crates/modelspec-runtime/src/dataset/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/init.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/mlm.rs
- confidence: cited
  path: spec/domains/data.yaml
- confidence: inferred
  path: spec/domains/training.yaml
- confidence: cited
  path: website/docs/reference/cli.md
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T21:36:58Z", actor: "human:timo", revision: 5}
- {from: "proposed", to: "active", at: "2026-10-24T21:36:58Z", actor: "human:timo", revision: 6}
---
# Pack token sequences and draw 80/10/10 MLM batches from Philox

## Context

`epic:bert-from-scratch` (In scope, Data: "sequence packing, the 80/10/10 MLM masking policy, and a
deterministic data order from the run seed"). `spec/domains/data.yaml` types
`modelspec.data.MaskingPolicy` (`selection_rate`, `mask_fraction`, `random_fraction`,
`keep_fraction`; "Draws use the Philox key/counter derivation (DESIGN.md § 2.3)"). DESIGN.md
§ 2.3 defines stream tags 0 to 2; `story:bert-dropout` adds tag 3.

## Open questions settled first

Two `UNMAPPED:` markers of `spec/domains/data.yaml` stand in front of this story, named by where
they sit and a quoted fragment of their text, never by line number. Each is settled before any
code, in that file (and in `spec/domains/training.yaml` where the second settlement puts the
record), with `ess specify validate --path .` in `spec/` passing afterwards. A settlement names its
source (the operator, DESIGN.md or another ESS document); where none answers, the story stops at
that step and reports the question instead of choosing.

1. Marker on `modelspec.data.MaskingPolicy`, text "the selection rate (the share of tokens selected
   for prediction)": the selection rate, and the Philox stream tag masking draws from. For the
   operator: Devlin et al. 2019 select 15 % of tokens; that is a candidate, not a source. The tag
   is the next free one after those of DESIGN.md § 2.3 at the time (4 once `story:bert-dropout`
   has landed), written into § 2.3.
2. The domain-level marker at the top of `spec/domains/data.yaml`, text "where a run records the
   corpus, tokenizer and masking policy it trained with": where a run records them (fields of
   `modelspec.training.TrainingRun`, data bindings of the artifact, or a new record), with the
   relations to `Dataset` and `Tokenizer` and their ownership, and whether masking is drawn per
   step from the run seed or fixed when the corpus is prepared. This story owns the type: it
   declares the run record in `spec/domains/` (`data.yaml` or `training.yaml`, as settled) with
   its relations, and generates its schema; `story:bert-train-data-binding` only writes values of
   that type.

## What this story decides

- `modelspec data pack --dataset <dir> --tokenizer <file> --seq-len <S> --out <dir>` encodes each
  split with `story:bert-tokenizer`, joins the documents in order with the separator token its
  settlement names (none if it names none), and cuts the stream into sequences of exactly `S`
  tokens; the partial tail of each split is
  dropped, so no padding is needed and every key of a packed sequence is unmasked. `key_mask`
  stays an input of the model, all ones in pre-training.
- A library function `mlm_batch(packed, split, policy, seed, step, batch_size)` returns
  `MlmBatch { ids, positions, key_mask, labels, weights }` for one step, flattened to `N = B * S`
  rows as `story:bert-attention` expects: `ids` with the 80/10/10 substitution applied, `labels`
  the original ids, `weights` 1 exactly at selected positions and 0 elsewhere, `positions` 0 to
  `S - 1` per sequence. Special tokens are never selected.
- Data order: epoch `e` visits the sequences of the `Train` split in a permutation drawn from its
  own Philox stream tag (the next free tag after the masking one, also written into § 2.3), keyed
  by the run seed; an incomplete last batch of an epoch is not used. The held-out split is read in
  stored order.
- `pack` also writes the held-out unigram baseline: the cross-entropy, in nats, of the
  non-special tokens of the `HeldOut` split under the token frequencies of the `Train` split with
  add-one smoothing. `story:bert-tiny-gpu-pretraining` measures its target against it.
- The packed record and its schema are typed in `spec/domains/data.yaml` before code (AGENTS.md,
  spec-first) and generated into `crates/modelspec-runtime/schema/`.

## Order on shared files

This story edits `spec/domains/data.yaml`, `crates/modelspec-runtime/src/dataset/`,
`crates/modelspec-runtime/schema/`, `crates/modelspec-cli/src/data.rs` and `src/lib.rs`, and
`website/docs/reference/cli.md` after `story:bert-tokenizer`, so it depends on it. It edits
`DESIGN.md` § 2.3 after `story:bert-dropout` and may edit `spec/domains/training.yaml` after
`story:bert-adamw-warmup`, so it depends on `story:bert-adamw-warmup` (which depends on
`story:bert-dropout`).

## Domain relations

- `MaskingPolicy` substitutes `modelspec.data.Vocabulary.mask_token_id` (the summary of
  `Vocabulary` in `spec/domains/data.yaml`: "mask_token_id names the token 80/10/10 masking
  substitutes"); `MlmBatch` ids stay in `[0, tokens.count)` by the same invariants.
- The draws use `modelspec.training.RngState.seed` of the run (`spec/domains/training.yaml`) and
  `modelspec.training.RngAlgorithm` `Philox4x32x10`.
- The relations of a run to its `Dataset`, `Tokenizer` and `MaskingPolicy` are not declared (the
  domain-level marker); this story declares them only as the second settlement states.

## Scope

- `spec/domains/data.yaml` (the two settlements, the packed record type), and
  `spec/domains/training.yaml` if the second settlement puts the record there;
  `crates/modelspec-runtime/schema/` (regenerated).
- `crates/modelspec-runtime/src/dataset/mlm.rs` (new), `dataset/mod.rs`,
  `src/train/init.rs` (the two stream tags).
- `crates/modelspec-cli/src/data.rs`, `src/lib.rs` (the `pack` verb).
- `crates/modelspec-runtime/tests/mlm.rs` (new), `crates/modelspec-cli/tests/data_pack.rs` (new).
- `DESIGN.md` § 2.3 (the masking and data-order derivation) and the revision line;
  `AGENTS.md` (Commands); `website/docs/reference/` through `task docs-generate`. No
  `CHANGELOG.md` line (`story:bert-train-data-binding` covers the data commands).

## Acceptance

`modelspec data pack` and `mlm_batch` turn the fixture dataset into fixed-length sequences and
reproducible 80/10/10 MLM batches at the settled selection rate, and `pack` records the held-out
unigram baseline.

Scenarios, each run by `task ci` through the command shown, except scenario 1:

1. `spec/domains/data.yaml` no longer contains the marker on `MaskingPolicy` ("the selection rate")
   or the domain-level marker ("where a run records the corpus, tokenizer and masking policy");
   typed declarations that cite their source stand in their place, among them the type of the run
   record,
   `cd spec && ess specify validate --path .` exits 0, and
   `grep -Eq "^\*\*Revision:\*\* [0-9.]+ \([0-9-]+\): [^;]*masking" DESIGN.md` exits 0.
2. `cargo test -p modelspec-cli --test data_pack sequences`: with `--seq-len 16` on the fixture,
   every sequence has 16 tokens, their concatenation equals the encoded token stream of the split
   without its dropped tail, the record validates against the generated schema, and two runs write
   identical bytes.
3. `cargo test -p modelspec-runtime --test mlm proportions`: over the batches of enough steps to
   cover `20_000` fixture positions, the selected share lies within `0.01` of the settled `selection_rate`, and among selected positions
   the mask, random and keep shares lie within `0.02` of `0.8`, `0.1`, `0.1`; no special token is
   selected; `labels` equal the original ids and `weights` are 1 exactly at selected positions.
4. `cargo test -p modelspec-runtime --test mlm derivation`: one `(seed, step)` gives an identical
   batch twice; the selection of element 0 equals the value the test derives from `philox4x32_10`
   with the settled tag; another step gives another selection.
5. `cargo test -p modelspec-runtime --test mlm epoch_order`: across one epoch every `Train`
   sequence appears in exactly one batch, at most once; two seeds give two orders.
6. `cargo test -p modelspec-cli --test data_pack baseline`: the baseline in the record equals the
   value the test computes from the fixture counts within `1e-9`.
7. `task docs-check` exits 0, and `website/docs/reference/cli.md` lists `data pack`.

## Out of scope

- Feeding batches to `modelspec train` (`story:bert-train-data-binding`).
- Padding, document boundaries inside the attention mask, and next-sentence pairs.
