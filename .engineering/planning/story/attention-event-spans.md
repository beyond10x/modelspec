---
format: aep.planning-md/3
id: story:attention-event-spans
kind: story
status: active
title: Attention over (start, length) spans, so the decider stores only real token rows
summary: 'Drops the decider''s 70.5 % padded event rows: about 279 to 160 ms per device step, same bits.'
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
- depends_on: story:attention-fewer-query-rows
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T12:36:22Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T12:36:22Z", actor: "human:timo", revision: 3}
---
## Outcome

`attention` accepts an `i32 [U, 2]` operand of (start, length) spans in place of the `f32` key
mask: each span is one sequence laid out back to back in the rows of q, k and v, keys are the
span's rows, and rows outside every span give `+0`. CPU interpreter, autodiff and Vulkan, forward
and the three gradients. A model whose sequences vary in length then stores only its real rows.

## Evidence that motivates it

The decider's event tokens are 70.5 % padding: blocks of 65,536 rows (256 events by 256
positions) hold a mean of 19,304 real tokens, median 19,145, p99 28,656, max 30,435 over 3,057
blocks of the 2026-10-29 run's data. Every mask keeps a prefix. Every layer of the event encoder
except attention works row by row, so a compact layout of 32,768 rows halves the rows those layers
compute; a read-only scoping (2026-10-29) estimated the device step at about 160 instead of 279 ms.
Padded rows add exact zeros today (masked keys; queries whose rows are never read), and the
matmul, LayerNorm-gamma and embedding gradients sum in one ascending order, so dropping the pad
rows while keeping the order keeps the bits apart from the sign of a zero.

Alternatives scoped and not taken: length classes (one program with two classes, about -28 %,
bits change, the model grows by about four copies of the encoder); packing several events per
sequence with segment ids (about -43 %, bits change because rows are reordered); compaction with
gathers around a padded attention (16 extra embedding gradients per step eat most of the saving).

## Acceptance

- CPU interpreter: attention over spans gives, for every span row, the bits of today's
  prefix-masked attention on the padded layout; rows outside every span give `+0`; overlapping,
  out-of-range or negative spans are refused with a named error.
- Gradient check of the span form on the CPU; the fewer-query-rows form (one query per span, its
  first row) works over spans.
- Vulkan: forward and the three gradients give the CPU bits on decider span shapes and keep
  today's bits and speed on BERT-tiny's mask form, with validation on.
- The decider on the compact layout (`tok_ids`, `tok_pos` of 32,768 rows, `evt_row` = span start,
  a token budget in batching) reaches the same weights hash as the padded layout after 20 steps,
  signs of zero aside, or the story records why not.
- Decider device step time, median of alternating A/B pairs at P0: lower by at least 30 %.
- BERT-tiny step time not worse (A/B, 200 steps).
