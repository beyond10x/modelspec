# Wave bert-1

Serves `vision:modelspec` and `epic:bert-from-scratch` (active 2026-10-24). Timo, 2026-10-24:
"when will we be able to train a transformer?" and, on the same day, "later is now :) - its your
project and you can make it a success. Take decissions by yourself!". Skill `aep:implementing`
0.23.2, wave mode; a second lane beside wave perf-1, one Opus implementor per lane, timed GPU
runs serialized between the lanes (decision D18 on `.engineering/waves/perf-1.md`).

## Selection

- The 17 BERT stories form one `depends_on` chain from the decomposition (each story depends on
  the one drafted before it) with two roots, `story:bert-corpus-splits` and
  `story:bert-embedding-lookup`, both of which depended on
  `story:observatory-saliency-gradient-x-input` for shared-file ordering only (their bodies say
  so: "I do not know whether a later one adds a dependency"). That edge is removed from
  `bert-corpus-splits` (D20); the observatory stories are not in flight.
- Order: the data chain first (`bert-corpus-splits`, `bert-tokenizer`, `bert-mlm-batches`), CPU
  only, no GPU; the operator chain (`bert-embedding-lookup` onward) runs beside it when the lane
  has room.
- Dispatch: `aep:implementor` (Opus) per unit, then `aep:adversary` (Opus) per green unit; the
  brief template of wave perf-1 (`~/.cache/modelspec-wave-bert-1/<unit>/brief.md`).

## Approval

Timo, 2026-10-24: "You have all authority now to make these decissions yourself. also, "later" is
now :) - its your project and you can make it a success. Take decissions by yourself!" (D17 on
`.engineering/waves/perf-1.md`), under the standing "I accept all you want, you act on behalf of
me" (`approval-record:perf-wave-1`).

## Units

| # | unit | branch | tree | build dir | scratch | commit | state |
|---|---|---|---|---|---|---|---|
| 1 | bert-corpus-splits | impl/bert-corpus-splits | wt-80e4bdccf4f3 | `target/` in the tree | `~/.cache/modelspec-wave-bert-1/corpus/scratch/` | 674c351, d4d3ccd | implemented: task ci exit 0 (1,736 passed, 0 failed, 168 ignored); adversary pass 1 red on the heading rule (29,444 against 28,475 on the real release), fixed by D22, 13 adversary cases green; ADV-CORPUS-2 drafted as story:content-hash-schema-constrained |

## Decisions taken by the coordinator

| id | decision | reason |
|---|---|---|
| D19 | The six `UNMAPPED:` markers of `spec/domains/data.yaml` that stand in front of the data chain are settled: corpus WikiText-103 raw release (Merity et al. 2016, CC BY-SA 3.0), `Dataset` holds raw text and has no relation to `Tokenizer`, a split is one file of the release (`Train` = `wiki.train.raw`, `HeldOut` = `wiki.valid.raw`) identified by role and `blake3` content hash, WordPiece lowercased with a vocabulary of 8,192, special tokens `[PAD]` 0 `[UNK]` 1 `[CLS]` 2 `[SEP]` 3 `[MASK]` 4, masking 0.15 selected and 0.8/0.1/0.1 (BERT § 3.1) drawn from Philox stream tag 3; the file-level marker on the run record waits for `story:bert-train-data-binding` | the story says "where none answers, the story stops"; the operator's grant of 2026-10-24 makes the coordinator the one who answers; each settlement takes the published BERT or WikiText-103 convention so results compare with published numbers |
| D20 | `story:bert-corpus-splits` no longer `depends_on story:observatory-saliency-gradient-x-input`; `story:bert-embedding-lookup` keeps its edge until it is dispatched | the edge recorded shared-file ordering against observatory stories that are not in flight; a dependency on undone work that nothing needs would hold the lane for nothing |
| D21 | A document of the raw release is an article starting at a top-level heading line `= <title> =` (section headings `= = <section> = =` do not count), text before the first heading one document; D19 said "blank-line-separated article" | the release separates paragraphs and headings by blank lines, not articles (first bytes of `wiki.valid.raw`: a blank line, ` = Homarus gammarus = `, a blank line, the first paragraph); the implementor proposed it, the coordinator checked the real release after downloading it to `~/.cache/modelspec-data/` (sha256 of the zip 91c00ae2…3794) |
| D22 | A document starts at a heading line ` = <title> = ` (not ` = = `) with a blank line directly before and directly after it; text before the first one is one document. On the real release: 28,472 train, 60 valid, 60 test. The published 28,475 is 3 higher; no heading rule the coordinator tried on the file explains the 3, so the spec records the strict count and notes the published one | ADV-CORPUS-1: the implementor's rule (any ` = … = ` line) counted 29,444 because table keys and equation fragments inside articles have that shape (967 of them with no blank line before; 5 with a blank line before and none after, all fragments; 0 with a blank line after and none before); probes by awk on `wiki.train.raw`, 2026-10-24 |

## Agent costs (Opus sub-agents; tokens from the task transcripts)

| agent | role | output tokens | input tokens | model calls |
|---|---|---|---|---|
