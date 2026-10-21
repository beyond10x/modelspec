---
format: aep.planning-md/3
id: review-result:bert-acceptance-round-1
kind: review-result
status: active
title: Plan critic acceptance, epic:bert-from-scratch, round 1
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

bert-vulkan-kernels-rowwise — the acceptance's tolerance is "the M1 tolerances", which no file holds as a number, so the device scenarios cannot be judged pass or fail until the body names a number or a settle-first step that produces one — .engineering/planning/story/bert-vulkan-kernels-rowwise.md:78

bert-vulkan-kernels-attention-loss-adamw — same defect: "within the M1 tolerances" is the pass criterion of scenarios 2 and 3, but `epic:m1-vulkan-execution` has no tolerance and no stories, and DESIGN.md says only "documented tolerances" — .engineering/planning/story/bert-vulkan-kernels-attention-loss-adamw.md:75

bert-adamw-warmup — scenario 6 passes whether `sgd` refuses schedule arguments or accepts them, and the "accepted and applied" branch has no check that the schedule is applied, so the scenario reads the same either way — .engineering/planning/story/bert-adamw-warmup.md:118

bert-heldout-evaluation — scenario `dropout_off` ("two evaluations of one parameter set give the same values bit for bit") names no difference from a run with dropout on, because a fixed token is also bit-reproducible; it should compare to the same fixture with the `dropout` call removed — .engineering/planning/story/bert-heldout-evaluation.md:79

bert-corpus-splits — scenario 5 ("The schema drift check of `task ci` covers the new schemas") is checked only by `grep -q crates/modelspec-runtime/schema Taskfile.yml` and `task ci` exiting 0, which says nothing about whether the check fails when a new schema drifts; it should name a drifted schema that makes it exit non-zero — .engineering/planning/story/bert-corpus-splits.md:111

What I read: 16 of 16 `story:bert-*` ids, each in full; `aep plan artifact kinds`, `lifecycle story`; `epic:m1-vulkan-execution`, `epic:bert-from-scratch`, the DESIGN.md revision line and § 9, and `Taskfile.yml`.

What I could not establish:
- Whether `epic:m1-vulkan-execution` will define the tolerances (draft, no stories).
- Whether any scenario's DESIGN.md `grep -Eq "^\*\*Revision:\*\* …"` still passes after a later story bumps the revision line.
- "each run by `task ci` through the command shown" is not true for the `ess specify validate` and DESIGN.md `grep` scenarios; `task ci` (`Taskfile.yml:63-70`) runs build, lint, test, `core-no-vulkan`, `check-examples` and `docs-check` only. Not counted as a finding.

```findings
- file: .engineering/planning/story/bert-vulkan-kernels-rowwise.md
  line: 78
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "the acceptance's tolerance is 'the M1 tolerances', which no file holds as a number, so the device scenarios cannot be judged pass or fail"
- file: .engineering/planning/story/bert-vulkan-kernels-attention-loss-adamw.md
  line: 75
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "'within the M1 tolerances' is the pass criterion of scenarios 2 and 3, but no file holds that tolerance"
- file: .engineering/planning/story/bert-adamw-warmup.md
  line: 118
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 6 passes whether sgd refuses schedule arguments or accepts them, and the accepted branch has no check that the schedule is applied"
- file: .engineering/planning/story/bert-heldout-evaluation.md
  line: 79
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario dropout_off names no difference from a run with dropout on; it should compare to the same fixture with the dropout call removed"
- file: .engineering/planning/story/bert-corpus-splits.md
  line: 111
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 5 checks only grep on Taskfile.yml and task ci exit 0; it should name a drifted schema that makes the check exit non-zero"
```
