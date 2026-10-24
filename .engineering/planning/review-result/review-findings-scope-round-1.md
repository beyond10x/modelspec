---
format: aep.planning-md/3
id: review-result:review-findings-scope-round-1
kind: review-result
status: active
title: Plan critic scope, review-finding stories, round 1
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
revision: 1
---
needs-revision

- story:matmul-shape-aware-tiling — finding 3 asks for "shape/hardware-aware tiling and autotune", and the body only claims shape (`m`, `k`, `n`); hardware-aware selection and autotune are neither claimed nor listed under Out of scope — review-0.7.1/REVIEW.md:7
- story:cpu-prepared-training-step — P4 says "`effects_in_order`, the `writes` table and `in_place_transposes` are rebuilt per step", and finding 7 asks to "reuse storage" instead of a "new Vec per op"; the outcome claims only verify-once, free-after-last-consumer and moved outputs, so neither the per-step table rebuild nor per-op allocation is claimed — review-0.5.0/REVIEW.md:55
- story:schedule-index-tables — finding 8 lists four lookups ("`TensorProgram::node()`", "`fusions()` scans for producers", "kernel dedup scans existing kernels", "parameter/view binding linear lookup") and P8 adds the O(S²) hazard pairs at `verify.rs:1313-1347`; the Why and outcome name only "node and consumer lookups", so kernel dedup, binding lookup and hazard pairs are neither claimed nor excluded — review-0.7.1/REVIEW.md:12

**What I read**
- 27 stories, each with `aep plan artifact show story:<id>` (dumped to one file).
- The two review files and `review-triage/TRIAGE.md`.
- `aep plan artifact graph`.
- `story:weight-grad-matmul-structural`, `story:input-readers-bound-before-allocating`, `story:train-refuses-non-finite-loss` and `story:m1-fuse-elementwise`.
- Spot greps at 8ffb903 (the draft commit, parent 496e98e) for C1, D1, O1, D2 and the P4 sites.

**Promises extracted and traced**
- I extracted 33 promises: 22 from review 1 (C2-C10, O2-O5, P1-P8, D2) and 11 from review 2 (findings 1-9, 11, 12).
- 33 of 33 trace to a story. Three are only partly claimed, which are the three findings above.
- The triage names five findings it does not file, and I treat all five as honest omissions: C1, O1, D1 (fixed), finding 10 (held by `story:weight-grad-matmul-structural`) and finding 5 (folded into `story:fast-math-mode`).
- No finding is claimed by two stories.
- No story reaches beyond its parent. The profile-numbers retake and the `task check` staleness rule both trace to the triage rows.

**What I could not establish**
- O1 (no `Access-Control-Allow-Origin` on `/events`) is listed as "Fixed" with no reason. `git grep -i -E "cors|allow-origin"` finds nothing at 8ffb903. `--ui` same-origin serving may be the reason, but I did not verify that, so the omission may hide a gap. The drafter should state why O1 is closed.
- Out of my lane: whether the acceptances are checkable, for example the 1.2x ratio in `story:parser-span-lookup-not-linear-in-column`. Also whether the dependency `story:vulkan-buffer-liveness-suballocation` depends_on `story:fuse-elementwise-chains-and-gradient-sums` serialises more than needed.

```findings
- file: review-0.7.1/REVIEW.md
  line: 7
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:matmul-shape-aware-tiling: finding 3 asks for 'shape/hardware-aware tiling and autotune' and the body claims only shape-dependent tiling; hardware-aware selection and autotune are neither claimed nor listed as out of scope"
- file: review-0.5.0/REVIEW.md
  line: 55
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:cpu-prepared-training-step: P4 says effects_in_order, the writes table and in_place_transposes are rebuilt per step and finding 7 asks to reuse storage instead of a new Vec per op; the outcome claims only verify-once, free-after-last-consumer and moved outputs, so neither is claimed or excluded"
- file: review-0.7.1/REVIEW.md
  line: 12
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:schedule-index-tables: finding 8 lists kernel dedup scans and parameter/view binding lookups and P8 the O(S^2) hazard pairs in verify.rs, while the body names only node and consumer lookups, so those sites are neither claimed nor excluded"
```
