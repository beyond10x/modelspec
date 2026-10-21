---
format: aep.planning-md/3
id: review-result:bert-scope-round-1
kind: review-result
status: active
title: Plan critic scope, epic:bert-from-scratch, round 1
relations:
- reviews: epic:bert-from-scratch
- reviews: story:bert-adamw-warmup
- reviews: story:bert-attention
- reviews: story:bert-corpus-splits
- reviews: story:bert-dense-gelu
- reviews: story:bert-dropout
- reviews: story:bert-embedding-lookup
- reviews: story:bert-heldout-evaluation
- reviews: story:bert-layer-norm-softmax
- reviews: story:bert-mlm-batches
- reviews: story:bert-mlm-cross-entropy
- reviews: story:bert-tiny-cpu-model
- reviews: story:bert-tiny-gpu-pretraining
- reviews: story:bert-tokenizer
- reviews: story:bert-train-data-binding
- reviews: story:bert-vulkan-kernels-attention-loss-adamw
- reviews: story:bert-vulkan-kernels-rowwise
revision: 1
---
needs-revision

epic:bert-from-scratch — "held-out MLM loss falls below a target" on the GPU run is promised, but no drafted item makes held-out evaluation (`--eval-every`, the `eval` lines) work on `--backend vulkan`. `story:bert-heldout-evaluation` is built on the CPU runner and `train-data-binding` is `--backend cpu --data`. `story:bert-vulkan-kernels-attention-loss-adamw` wires only `--data` for Vulkan, and `story:bert-tiny-gpu-pretraining` assumes `--eval-every` already exists there. The natural owner is `story:bert-vulkan-kernels-attention-loss-adamw`, or `story:bert-heldout-evaluation` extended to the Vulkan runner. — .engineering/planning/epic/bert-from-scratch.md:57 ("A BERT-tiny `.mspec` passes `modelspec check` and trains on the GPU backend; held-out MLM loss falls below a target"), against .engineering/planning/story/bert-tiny-gpu-pretraining.md:21-22 and .engineering/planning/story/bert-vulkan-kernels-attention-loss-adamw.md:30-31, 55

epic:bert-from-scratch — "the run is watched in the training observatory" is promised, and no item claims it for the pre-training run. `story:bert-heldout-evaluation` streams metric samples only for a CPU run (scenario 4). `epic:training-observatory` puts GPU out of its scope ("the observatory must run against `--backend cpu`"). `story:bert-tiny-gpu-pretraining` names no serve flag and no observatory view in its acceptance. The natural owner is `story:bert-tiny-gpu-pretraining`, or an explicit "not covered, operator to decide" line in the epic or the drafter's report. — .engineering/planning/epic/bert-from-scratch.md:21 ("the run is watched in the training observatory"), against .engineering/planning/story/bert-tiny-gpu-pretraining.md:74-82

What I read: 18 artifacts: the epic, the 16 story files, the m1 and observatory epics and `story:observatory-live-event-stream`; `aep plan artifact graph`; `grep UNMAPPED` in `spec/domains/` and `grep bert` in `DESIGN.md`.

- Promises extracted: 29 from the epic. Traced to an item: 27. The two findings above are the rest.
- Open markers: the 8 `UNMAPPED:` markers named in the epic's Specification section each have exactly one owning story.
- Named omissions, not gaps: the optional BERT-mini/small and downstream probe; attention-probability dropout; parameter groups for weight decay; the numeric loss target, replaced by the unigram baseline with the reason given.
- No reach beyond the epic found.
- DESIGN.md: the promised M3 row and iteration 4 update is present (revision 0.7).
- Parameter count: `story:bert-tiny-cpu-model`'s 4,367,290 recomputed and correct.

What I could not establish: several stories quote epic text that is not verbatim (headings "Target:" and "Model:" the epic does not have; it has "Execution:" and "Model size:"), e.g. `story:bert-tiny-cpu-model` lines 18-21, `story:bert-vulkan-kernels-rowwise` line 19, `story:bert-adamw-warmup` line 18. Each quoted sentence still traces to a real epic sentence, so the verdict does not rest on it.

```findings
- file: .engineering/planning/epic/bert-from-scratch.md
  line: 57
  category: scope
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "held-out MLM loss on the GPU run is promised, but no drafted item makes held-out evaluation (--eval-every, eval lines) work on --backend vulkan; natural owner story:bert-vulkan-kernels-attention-loss-adamw"
- file: .engineering/planning/epic/bert-from-scratch.md
  line: 21
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "'the run is watched in the training observatory' is promised and no item claims it for the pre-training run; natural owner story:bert-tiny-gpu-pretraining, or a named omission"
```
