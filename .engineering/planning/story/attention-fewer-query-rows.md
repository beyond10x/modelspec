---
format: aep.planning-md/3
id: story:attention-fewer-query-rows
kind: story
status: active
title: Attention with fewer query rows than key rows
summary: 'Lets the decider compute its last encoder layer on the one row per block it reads: about 23 % of the step.'
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T11:50:57Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T11:50:58Z", actor: "human:timo", revision: 3}
---
## Outcome

`attention(q, k, v, mask, heads = n)` accepts fewer query rows per sequence than key rows: q has
`B * Lq` rows, k and v have `B * Lk` rows, the mask stays `[B, Lk]`-shaped per sequence, and
`Lq` divides into the query rows. On the CPU interpreter, in autodiff and on Vulkan, forward and
the three gradients. A model that reads one row per sequence after its last encoder layer can
then compute that layer's queries, residual, LayerNorm and feed-forward on those rows only.

## Evidence that motivates it

The decider model (`models/decider.mspec`, decider repository) reads `t4` only through
`embedding(t4, evt_row)` with `evt_row = u * 256` (decider `src/batch.rs:497`): one of the 256
positions of every block. Its encoder layer 3 still computes all 65,536 rows of `eq3`, the
attention output, `eh3`, the feed-forward and `t4`. A read-only scout estimated about 308 GFLOP of
forward and backward work per step, 23 % of what the step computes (scouts review, 2026-10-29).
Keys and values must stay at every row: each selected query attends to its whole block.

## Acceptance

- CPU interpreter: `attention` with `Lq = 1` and with `Lq = Lk` (today's case); the `Lq = 1`
  result for each sequence equals row 0 of the `Lq = Lk` result in bits, forward and gradients.
- Gradient check of the new form on the CPU.
- Vulkan: forward and the three gradients on `Lq = 1` decider shapes equal the device's own
  `Lq = Lk` rows in bits and the CPU within the GPU tolerance (the device's exp and division are
  not correctly rounded); `Lq = Lk` BERT-tiny shapes keep their bits; validation on.
- A decider model rewritten to gather `t3` at `evt_row` before layer 3 trains to a loss within the
  spread of a reorder-only control over 20 steps: today's model with two independent lines swapped,
  which changes only the order gradients are summed in (SGD at lr 0.025 amplifies that rounding;
  the gate-5 run tolerance holds through step 15 for both and for neither at steps 16-20).
- Decider device step time, median of alternating A/B pairs at P0: lower by at least 15 %.
- BERT-tiny step time not worse (A/B, 200 steps).
