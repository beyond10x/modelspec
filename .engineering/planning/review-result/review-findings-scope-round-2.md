---
format: aep.planning-md/3
id: review-result:review-findings-scope-round-2
kind: review-result
status: active
title: Plan critic scope, review-finding stories, round 2
relations:
- reviews: story:checkpoint-reader-refuses-out-of-order-records
- reviews: story:checkpoint-size-checks-never-wrap
- reviews: story:cpu-finite-check-updated-params-only
- reviews: story:cpu-matmul-output-skips-zero-fill
- reviews: story:cpu-prepared-training-step
- reviews: story:crates-page-names-current-release
- reviews: story:dense-step-reuses-batch-buffers
- reviews: story:fast-math-mode
- reviews: story:float-literals-round-once
- reviews: story:fuse-elementwise-chains-and-gradient-sums
- reviews: story:matmul-shape-aware-tiling
- reviews: story:mean-mse-adjoints-divide-by-n
- reviews: story:observatory-accepts-bare-lf-requests
- reviews: story:observatory-emit-copies-outside-hub-lock
- reviews: story:observatory-ended-stream-answers-204
- reviews: story:param-init-refuses-oversized-allocation
- reviews: story:parser-span-lookup-not-linear-in-column
- reviews: story:profile-begin-stamp-after-barrier
- reviews: story:schedule-index-tables
- reviews: story:serve-ends-without-signal-handler
- reviews: story:unclosed-train-reports-missing-brace
- reviews: story:verifier-counts-every-parameter-write
- reviews: story:vulkan-buffer-liveness-suballocation
- reviews: story:vulkan-loss-slots-written-by-kernel
- reviews: story:vulkan-pipeline-cache
- reviews: story:vulkan-scalar-operand-in-reduce-and-loss
- reviews: story:vulkan-train-compiles-once
- reviews: story:vulkan-step-counters
revision: 1
---
approve

All three of my round-1 findings are resolved, and no new finding needs a revision. Two notes remain; they do not set the verdict.

**Round-1 findings**
| story | round-1 finding | now |
|---|---|---|
| matmul-shape-aware-tiling | hardware-aware tiling and autotune neither claimed nor excluded | resolved: excluded with a reason at `.engineering/planning/story/matmul-shape-aware-tiling.md:70`. DESIGN.md:603 is "Autotune storage", an open question in § 12 (DESIGN.md:595). |
| cpu-prepared-training-step | P4 table rebuild and finding 7 storage reuse not claimed | resolved: claimed in the outcome and in **tables-once-per-run** (`:38`) and **storage-reused** (`:44`) |
| schedule-index-tables | kernel dedup, binding lookups and hazard pairs not claimed | resolved: claimed at `:26-29`. At HEAD, `schedule.rs:569-573` is the kernel dedup, `:304`, `:597` and `:605` are the linear binding searches, and `verify.rs:1313-1347` is the step closure. |

**Notes (new, approve)**
- story:fuse-elementwise-chains-and-gradient-sums: finding 2 suggests "fuse contiguous elementwise DAGs incl. backward" (review-0.7.1/REVIEW.md:6). The outcome says "chains" (`.engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md:45`). That matches the defect as the review states it, so this is only a note. A fan-out DAG would have to be claimed or excluded by name.
- story:vulkan-step-counters: its two consumers read only `--stats`. **profile-lines** (`.engineering/planning/story/vulkan-step-counters.md:46`) also adds three `--profile` lines that neither consumer asks for. Review-0.7.1/REVIEW.md:19 ("GPU dispatches/barriers/bytes/VRAM per step") would cover them, but the Why (`:24`) cites only critic round 1.

**What I read**
- Read first: `review-0.5.0/REVIEW.md`, `review-0.7.1/REVIEW.md` and `review-triage/TRIAGE.md`.
- All 28 stories, through `aep plan artifact show`.
- My round-1 report, `r1-outcomes.md`, and the list of open stories in the store.
- Code and docs at HEAD 2b17433: `DESIGN.md` § 12, `schedule.rs`, `verify.rs:1305-1350`, and `checkpoint.rs:38-48`. The E-CKPT-002 text there already covers "manifest size runs past the end", so **declared-size-past-file** is a regression test of existing behaviour, not new reach.
- Also read: `story:input-readers-bound-before-allocating`, which covers file bounds, not declared shapes, so it does not overlap the C5 story.

**Promises traced:** 33 extracted (22 from review 1, 11 from review 2). All 33 trace to a story and none is now partial.
- The settled items are honest omissions: C1, D1, O1, #10 and #5.
- Settled operator decisions, not findings: ADR-008 and the mean-mse wording-only story.
- No finding is claimed by two stories.
- vulkan-step-counters traces to no finding. It says so and serves findings 1 and 2.

**What I could not establish**
- Review-0.7.1/REVIEW.md:18-19 (suggested order and perf suite) are suggestions, not numbered findings, and the triage does not mention them. Two items in them are unclaimed: RSS and compile time at 100 and 1000 nodes. I did not count them as promises.
- Out of my lane: whether the acceptances are checkable, and whether the `depends_on` edges serialise more than needed.

```findings
- file: .engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md
  line: 45
  category: scope
  severity: note
  verdict: approve
  origin: pre-existing
  message: "finding 2 suggests fusing contiguous elementwise DAGs; the outcome claims chains. That matches the defect as stated, but non-chain DAGs (fan-out inside a group) are neither claimed nor excluded by name"
- file: .engineering/planning/story/vulkan-step-counters.md
  line: 46
  category: scope
  severity: note
  verdict: approve
  origin: introduced
  message: "profile-lines adds --profile counters that neither consumer story reads; review-0.7.1 line 19 (GPU barriers/bytes/VRAM per step) would trace them, but the Why cites only critic round 1"
```
