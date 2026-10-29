---
format: aep.planning-md/3
id: epic:bert-from-scratch
kind: epic
status: implemented
title: 'Milestone: pre-train a small BERT from scratch in ModelSpec'
summary: A BERT-tiny class encoder written in .mspec, pre-trained with masked-language modelling on the GPU backend to a stated loss target.
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- informed_by: epic:training-observatory
- informed_by: executable-system-specification:modelspec-contracts
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T18:51:35Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-24T18:51:35Z", actor: "human:timo", revision: 5}
- {from: "active", to: "implemented", at: "2026-10-29T20:34:02Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"test_result":1,"metric_observation":1}}}
---
# Milestone: train a small BERT from scratch, written in ModelSpec

## Outcome

A BERT-style bidirectional encoder, written as one or a few `.mspec` files, is pre-trained from
random initialisation with masked-language modelling on a small public corpus, and its loss falls
to a stated target; the run is watched in the training observatory.

## Operator intent

"some milestone: train something like a BERT from scratch, implemented in mspec" (operator,
2026-10-21).

## Relation to DESIGN.md

DESIGN.md § 1 row M3 and § 11 iteration 4 named a small GPT-like model as the M3 proof. Decided by
the operator on 2026-10-21 (option A): BERT joins the GPT-like model as a second M3 proof model, so
M3 adds bidirectional (unmasked) attention, GELU and an MLM head beside causal masking. DESIGN.md
row M3 and iteration 4 are updated in the same change as this epic.

## In scope

- Language and operators: embedding lookup, positional embeddings, layer norm, softmax, scaled
  dot-product multi-head attention with an attention mask, GELU, dropout with the existing Philox
  key/counter derivation, cross-entropy over a vocabulary, AdamW with weight decay and warmup.
- Data: a WordPiece (or byte-level) tokenizer and vocabulary, sequence packing, the 80/10/10 MLM
  masking policy, and a deterministic data order from the run seed. Tokenizer and data pipeline in
  Rust.
- Model size: BERT-tiny class first (2 layers, hidden 128, 2 heads, vocabulary about 30k,
  sequence 128; about 4.4M parameters, to verify against Turc et al. 2019, arXiv 1908.08962), then optionally
  BERT-mini/small.
- Execution: the GPU backend (epic:m1-vulkan-execution) for the full run; the CPU interpreter for
  gradient checks on a 2-layer, hidden-32 configuration.
- Evaluation: MLM loss and accuracy on a held-out split; optionally one downstream probe task.

## Out of scope

- Matching published BERT-base scores; distributed training; mixed precision beyond what M3 adds.

## Acceptance

- Every new operator passes the f64 finite-difference gradient check on the CPU interpreter.
- A BERT-tiny `.mspec` passes `modelspec check` and trains on the GPU backend; held-out MLM loss
  falls below a target set in decomposition from a reference run.
- Two runs with one seed produce identical loss curves on the CPU configuration.

## Dependencies

- epic:m1-vulkan-execution (GPU runtime). Device runs are the operator's to start.
- epic:training-observatory for watching the run (not for correctness).

## Specification

The nouns are modelled in `spec/domains/data.yaml` (Tokenizer with its Vocabulary type, Dataset,
the masking policy) and `spec/domains/training.yaml` (AdamW in `OptimizerKind`,
`LearningRateSchedule`). `ess specify validate --path .` in `spec/`: `modelspec v1 — 7 file(s),
valid`. Open questions are the 8 `UNMAPPED:` markers in those two files (tokenizer kind, special
tokens, splits, corpus, masking rate and stream, schedule); a story that depends on one settles it
first.
