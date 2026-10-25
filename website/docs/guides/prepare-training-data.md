---
title: Prepare training data
description: Turn a WikiText-103 raw release into packed token sequences with modelspec data prepare, data tokenizer and data pack, and train a masked-language model on them.
---

# Prepare training data

A masked-language model trains on token sequences, not on raw text. Three commands turn a corpus
into a packed directory that `modelspec train --data` reads, and each writes a record whose id the
next one carries, so a training run names exactly which text, tokenizer and sequence length it saw.

| Command | Reads | Writes |
|---|---|---|
| `modelspec data prepare` | a WikiText-103 raw release: `wiki.train.raw` and `wiki.valid.raw` | the `Train` and `HeldOut` splits, `train.raw` and `heldout.raw`, and their record `dataset.json` |
| `modelspec data tokenizer` | the `Train` split | a WordPiece tokenizer of an exact vocabulary size, the five special tokens included, as one JSON record |
| `modelspec data pack` | both splits and the tokenizer | `train.tokens` and `heldout.tokens`, sequences of exactly `--seq-len` tokens, their binding `binding.json` and their record `packed.json` |

Nothing is downloaded. The commands below run from the root of a checkout on the small corpus the
tests use, `crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`, which has the layout of a
release; point `data prepare` at an unpacked release to prepare the real one. Each output below
was printed by the command line built from `main`. The three `data` commands are in 0.10.0;
training on a packed directory (step 4) is on `main` and not yet released.

## 1. Split the corpus

```sh
modelspec data prepare crates/modelspec-runtime/tests/fixtures/corpus-tokenizer --out dataset
```

```text
dataset eecbf2be-db36-8673-ac07-792cc3366c87 (corpus-tokenizer)
Train 4 documents, 1453 bytes, blake3 6c4a92c4790746e1a983cfd54656ab402538f58eccef2ee04de463e4bc653fbd
HeldOut 1 documents, 324 bytes, blake3 3433fa3593ed4836ea4d27a5072ba1f572bf5b60c64348dd55d38c3b4ec39610
```

`wiki.train.raw` becomes the `Train` split and `wiki.valid.raw` the `HeldOut` split, copied as
they are. The dataset id is derived from the content, so the same release gives the same id on
any machine.

## 2. Train the tokenizer

```sh
modelspec data tokenizer --dataset dataset --vocab-size 300 --out tokenizer.json
```

```text
tokenizer e337abc4-7dc9-8146-80be-736ae6f83a8e (300 tokens)
```

Only `train.raw` is read, so nothing of the `HeldOut` split leaks into the vocabulary. The
vocabulary has exactly `--vocab-size` tokens.

## 3. Pack the sequences

```sh
modelspec data pack --dataset dataset --tokenizer tokenizer.json --seq-len 16 --out packed
```

```text
packed 2bcf89dc-61ba-8c79-bff4-10a47b648b1d (seq_len 16)
Train 26 sequences, blake3 b243d1b9ee16d1811ca83578e0ad0c433d782ba31a19b4868aae4f01fd25c83d
HeldOut 9 sequences, blake3 dd541e3eaa3999dd8a47863bf222b6c7b915b88ab31e70384cb0f08451ec48f3
binding 06307131-1d38-832c-8f2a-4f500bd3d35d (dataset eecbf2be-db36-8673-ac07-792cc3366c87, tokenizer e337abc4-7dc9-8146-80be-736ae6f83a8e)
heldout unigram baseline 5.566830996867244 nats
```

Both splits are encoded and cut into sequences of 16 tokens; the partial tail of each split is
dropped. The held-out unigram baseline is the mean loss, in nats, of predicting each held-out token
that is not special by its add-one-smoothed frequency in `Train`: a masked-language model has
learned something once its held-out loss is below it. `binding.json` ties the dataset, the tokenizer, the sequence length and the masking
policy together. Its fields, from the run above:

| field | value |
|---|---|
| `binding_id` | `06307131-1d38-832c-8f2a-4f500bd3d35d` |
| `dataset_id` | `eecbf2be-db36-8673-ac07-792cc3366c87` |
| `tokenizer_id` | `e337abc4-7dc9-8146-80be-736ae6f83a8e` |
| `seq_len` | 16 |
| `masking_policy.selection_rate` | 0.15 |
| `masking_policy.mask_fraction` | 0.8 |
| `masking_policy.random_fraction` | 0.1 |
| `masking_policy.keep_fraction` | 0.1 |
| `state` | `Recorded` |

## 4. Train on the packed directory

Save the test model
[`mlm_bag.mspec`](https://github.com/beyond10x/modelspec/blob/main/crates/modelspec-cli/tests/fixtures/mlm_bag.mspec),
a token table tied as its own decoder ([BERT operators](../concepts/bert-operators.md#a-model-that-uses-them)
shows it). Its `train` block declares `ids` as the input and `labels` and `weights` as targets,
each a tensor of the masked-language-model batch, 64 tokens or 4 sequences of 16 per step. The
`init` and `param` lines hold 4,800 numbers each, so this run is cut to 72 columns:

```sh
modelspec train mlm_bag.mspec --backend cpu --data packed | cut -c 1-72
```

```text
data dataset eecbf2be-db36-8673-ac07-792cc3366c87 tokenizer e337abc4-7dc
init @MlmBag.table: Tensor<f32, [300, 16]> = [-0.11207246, -0.03667358, 
step 0 loss 5.7054143
step 50 loss 4.766625
step 100 loss 5.1877227
step 150 loss 5.508473
step 200 loss 4.231307
param @MlmBag.table: Tensor<f32, [300, 16]> = [-0.07778824, -0.02267372,
```

The first line, in full, names the binding the run trained on:

```text
data dataset eecbf2be-db36-8673-ac07-792cc3366c87 tokenizer e337abc4-7dc9-8146-80be-736ae6f83a8e seq_len 16 masking 0.15/0.8/0.1/0.1 binding 06307131-1d38-832c-8f2a-4f500bd3d35d
```

Step `k` feeds the `Train` batch of step `k`, drawn from the run's seed: the sequences in a seeded
epoch order, 15% of their tokens selected, and of those 80% replaced by the mask token, 10% by a
random token and 10% left as they are. The labels are the original ids, and the weights are 1 at
the selected tokens and 0 elsewhere. `--record` adds the binding's id to the run's `RunStarted`
event as `data_binding_id`.

A declaration that names no batch tensor is `E-DATA-005`, one of the wrong type or shape
`E-DATA-006` ([Diagnostics](../reference/diagnostics.md#training-data)). A packed `--data` runs on
`--backend cpu` only, and `--eval` beside it is a usage error.

:::caution[Planned]

Held-out evaluation during training, the loss and accuracy on the `HeldOut` split at a fixed
interval, is planned. It is not built: the `HeldOut` sequences are packed but no command reads them
yet.

:::
