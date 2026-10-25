---
format: aep.planning-md/3
id: review-result:adversary-tiny-cpu-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-tiny-cpu-model
relations:
- reviews: story:bert-tiny-cpu-model
revision: 1
---
unit: story:bert-tiny-cpu-model (worktree wt-4b60fa8840ab, head 9cc6b69)
verdict: findings 6 (blocker 0, warning 4, note 2)
cases: executed 5→17, red 3
origin: introduced 5 / pre-existing 1 / undecided 0
wrote-outside-worktree: `<wave-root>/tiny-cpu/adv1-scratch/` (scratch copy with its target cleaned, mutants, packed probes, a WikiText-103 excerpt)
needs-coordinator: ADV-TINY-2: D19/D23 fixed the corpus tokenizer at 8,192 and said story bodies that say 30522 are amended when dispatched; the story and D70 dispatched V = 30522

## Cases added

| file | cases | now |
|---|---|---|
| `crates/modelspec-runtime/tests/adversary_bert_cpu.rs` | 6: a walk of the Tensor IR of both files (embedding sum, LN, dropout 0.1; per layer q/k/v from the layer input, `Attention{heads: 2}` with `key_mask`, the residual `add`, post-LN, gelu feed-forward; head dense, gelu, LN, `transpose(tok_table)`, `mlm_bias`, cross entropy; 5 dropouts) and its rejection of 6 in-memory mutants; dropout active at the gradient-check point; a finite key-bias move leaves the masked-batch loss (the `l0_bq` control moves it); the checked table elements are spread and reach embedded rows; embedded rows match central differences | 6 green |
| `crates/modelspec-cli/tests/adversary_bert_tiny_cli.rs` | 6: another seed prints other `step` lines; the twin differs from the example only in shapes, `steps` and `warmup_steps` (10 % in both); that comparison refuses heads, p, lr, weight_decay, seed and layer-index mutants; ADV-TINY-1, -2, -3 | 3 green, 3 ignored and red |

## Red cases (verbatim)

```
adv_tiny_1 panicked at adversary_bert_tiny_cli.rs:138:5:
held-out loss 5.617779 at eval 0, 5.6028047 at eval 500; the unigram baseline is 5.482577090758218; ln(300) = 5.703782474656201
adv_tiny_2 panicked at adversary_bert_tiny_cli.rs:316:5:
assertion `left == right` failed: tok_table rows of examples/bert_tiny.mspec  left: 30522 right: 8192
adv_tiny_3 panicked at adversary_bert_tiny_cli.rs:337:5:
the 300-row twin trained on a 200-token packed directory: step 0 loss 5.816008 / step 50 loss 5.7529655 / step 60 loss 5.6262946  left: Some(0) right: Some(0)
```

## Suite run after the cases existed (`--include-ignored`)

| target | result | exit |
|---|---|---|
| bert_tiny | 3 passed | 0 |
| gradients_bert_cpu | 1 passed, 18.54 s | 0 |
| bert_cpu_train | 1 passed, 37.44 s | 0 |
| adversary_bert_cpu | 6 passed | 0 |
| adversary_bert_tiny_cli | 3 passed, 3 failed (adv_tiny_1, 2, 3) | 101 |

Clippy `-D warnings` and rustfmt clean on both new targets.

## Mutants (scratch copy; "both" = example and twin)

| mutant | unit suite (5 tests) | killed by |
|---|---|---|
| a: heads = 1 in both | survived | the structure walk |
| b: layer-1 attention residual dropped, both | survived | the structure walk |
| c: position offset | n/a (`positions` comes from `mlm_batch`; the walk asserts `pos_table` is indexed by it) | — |
| d: weight_decay 0.0, both | survived | survived (all adversary cases too) |
| e: decoder bias removed, both | `parameter_count` | + the structure walk |
| f: warmup_steps 0, both | survived | `twin_differs_…` (10 % rule) |
| g: seed ignored | not mutable without the implementation | `another_seed_…` shows the seed is read |
| h: lr 1e-9, twin only | survived (`reproducible` passes: 5.464933 → 5.2910175, eval flat 5.617779) | `twin_differs_…`, adv_tiny_1 |
| i: heads = 1, twin only | survived | the structure walk, `twin_differs_…` |
| j: gelu → relu in the MLM head, both | survived | the structure walk |
| k: p = 0.5, twin only | survived | the structure walk, `twin_differs_…` |

## Findings

| id | file:line | verdict / origin | what was measured | what reaches it |
|---|---|---|---|---|
| ADV-TINY-1 | bert_cpu_train.rs:152 | NEEDS-CHANGE / introduced | `last < first` compares two 2-sequence batches, so noise decides it, and mutant h passes; the real run's held-out loss goes 5.6178 → 5.6028 with accuracy 0, never below the 5.4826 unigram baseline; lr 1e-4 over 500 steps caps the Adam movement near 0.05 | scenario 4 and the status row |
| ADV-TINY-2 | examples/bert_tiny.mspec:12 | CONFIRMED / introduced | V = 30522 against the repository's only corpus tokenizer (8,192; D19/D23, data.yaml `TokenizerKind`): 22,330 rows (2.88M of 4.37M parameters) are never read; the example on 8192-packed data exits 0 | `story:bert-tiny-gpu-pretraining` trains on that tokenizer |
| ADV-TINY-3 | crates/modelspec-runtime/src/train/data.rs:848 | CONFIRMED / pre-existing | `bind` does not compare the packed `vocabulary_size` with the table rows; a wider table trains silently, a narrower one fails at step 0 | the same path as ADV-TINY-2 |
| ADV-TINY-4 | crates/modelspec-cli/tests/bert_tiny.rs:118 | NEEDS-CHANGE / introduced | `shape_of_source` masks every digit, so mutants h, i and k pass `twin_is_the_example_with_other_sizes` | AGENTS.md cites this test as the guard |
| ADV-TINY-5 | crates/modelspec-cli/tests/bert_tiny.rs:135 | NEEDS-CHANGE / introduced | nothing in the unit's suite looks at the architecture: a, b and j survive all 5 tests | any edit to either file |
| ADV-TINY-6 | examples/bert_tiny.mspec:7 | INFEASIBLE / introduced | the comment says "AdamW with BERT's settings"; BERT excludes LN and bias from decay, initialises N(0, 0.02), has attention-probability dropout, warms up 1 %; here `pos_table` xavier std 0.088 is 11× `tok_table`'s 0.0081 | the language has only xavier, zeros and ones, no decay exclusion, no attention dropout |

## Attacked, not broken

Residuals, post-LN order, 2 heads, gelu feed-forward, MLM head order, embedding LN before dropout, `key_mask`, tied decoder (all in the IR); count 4,367,290 and 41 unique paths; the key-bias zero-gradient argument under a non-uniform mask and a finite move; dropout active in the gradient check with the same tokens for the gradient and both differences; the checked indices spread and reaching embedded rows; the seed is read; eval runs without dropout; docs rows; the example trains (a copy with `warmup_steps = 1`, 2 steps, exit 0, 885 s in a debug build with concurrent builds; step 0 loss 10.331078 against ln 30522 = 10.3264).

```findings
- file: crates/modelspec-cli/tests/bert_cpu_train.rs
  line: 152
  category: mutant
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: reproducible's last-below-first check passes for a twin at lr 1e-9 that does not learn, and the real twin's held-out loss falls 0.015 nats and never reaches the unigram baseline.
- file: examples/bert_tiny.mspec
  line: 12
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: The example declares V 30522 where D19/D23 and spec data.yaml fix the corpus tokenizer at 8,192, leaving 2.88M of its 4.37M parameters unreachable on the repository's data.
- file: crates/modelspec-runtime/src/train/data.rs
  line: 848
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: bind does not compare the packed vocabulary_size with the token table rows, so a table wider than the vocabulary trains without a word.
- file: crates/modelspec-cli/tests/bert_tiny.rs
  line: 118
  category: mutant
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: shape_of_source masks every digit, so a twin with heads 1, dropout 0.5 or lr 1e-9 still passes twin_is_the_example_with_other_sizes.
- file: crates/modelspec-cli/tests/bert_tiny.rs
  line: 135
  category: mutant
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: No unit test observes the architecture; a dropped residual, one head or relu in the MLM head made to both files survives all five unit tests.
- file: examples/bert_tiny.mspec
  line: 7
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: The comment claims BERT's AdamW settings, but decay applies to LN and bias, init is xavier (pos_table std 11x tok_table's), there is no attention dropout, and warm-up is 10 % where BERT pre-training used 1 %; the language offers no alternative.
```
