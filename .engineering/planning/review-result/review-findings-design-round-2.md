---
format: aep.planning-md/3
id: review-result:review-findings-design-round-2
kind: review-result
status: active
title: Plan critic design, review-finding stories, round 2
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
needs-revision

- story:matmul-shape-aware-tiling — its outcome picks a new tile for every matmul site, k11 included. It runs after story:profile-begin-stamp-after-barrier, which changes how every `--profile` kernel time is measured. story:weight-grad-matmul-structural's acceptance 1 (k11 at most 4.0 µs from `--profile`) and its "against the default tile" comparison read both of those figures, and no `blocks` edge from this story to story:weight-grad-matmul-structural, nor any `depends_on` edge the other way, records that order. To resolve: add `blocks: story:weight-grad-matmul-structural` to this story, since it already depends on the stamp story — .engineering/planning/story/matmul-shape-aware-tiling.md:34-35, .engineering/planning/story/profile-begin-stamp-after-barrier.md:25, .engineering/planning/story/weight-grad-matmul-structural.md:25,32-33,36-37

**Round-1 findings**

| # | Round-1 finding | Now |
|---|---|---|
| 1 | mean-mse needs a `Div` op | resolved: the story now changes only DESIGN.md and adds a test (mean-mse-adjoints-divide-by-n.md:52) |
| 2 | weight-grad's pinned sha vs mean-mse | resolved: the gradient bits no longer change |
| 3 | weight-grad's k11 figures vs tiling and stamp | stands, restated above on a story in the set. The round-1 rebuttal (D14) checked whether the two stories' scopes share a file. This finding is about an acceptance line reading a figure another story changes, and such pairs usually share no file |
| 4 | fuse barrier counter has no producer | resolved: story:vulkan-step-counters, with an edge (fuse-elementwise-chains-and-gradient-sums.md:9,52) |
| 5 | liveness peak-bytes and allocation counters have no producer | resolved: same story, with an edge (vulkan-buffer-liveness-suballocation.md:10,49-53) |
| 6 | fast-math and loss-slots split one loss kernel | resolved: edge added, and both stories say who owns what (fast-math-mode.md:9,47-48; vulkan-loss-slots-written-by-kernel.md:43) |

New: nothing.

**What I read**
- All 28 stories in full (at HEAD 132f661), plus story:weight-grad-matmul-structural and the k11 verification report.
- My round-1 report and the round-1 outcomes.
- Commands: `aep plan artifact show`, `relations`, `validate` (reports valid), `graph --format json`, `waves --status draft --format json`.
- Edges: the set declares 10 (6 `depends_on`, 4 `informed_by`). I also walked all 118 `depends_on` and `blocks` edges in the store through `tsort`: no cycle.
- The store's non-ordering edges do form one loop (verification-report:mnist-benchmark-cuda-baseline and epic:pytorch-10x-mnist), but it lies outside the set and uses no ordering edge.
- Chains: counters → fuse → liveness, and stamp → tiling → fast-math, with loss-slots → fast-math. Nothing serialises the set.
- Acceptance reads: I traced 12.
  - 7 read something a story in the set produces. 6 of those have an edge, directly or through another story; the 7th is the k11 read above.
  - 5 read things that already exist: `pipeline_creation_us` (profile.rs:264), `compile_us` (profile.rs:267), `cooperative_matrix` (device.rs:84), `numerics_class` (runner.rs:1071), and the `evaluate_step` inputs (optimizer.rs:121-125, so the CPU allocation count leaves out the batch feed).

**What I could not establish**
- Whether liveness's aliasing check (`verifier-checks-aliasing`) goes through the buffer hazard closure that story:schedule-index-tables rewrites (verify.rs:1308-1340). If it does, the two stories hold halves of one abstraction.
- Out of my lane (parallel safety):
  - fast-math-mode, vulkan-train-compiles-once and serve-ends-without-signal-handler all edit `crates/modelspec-cli/src/train.rs` with no order between them. The `--fast-math` flag has to pass through the compile call at train.rs:372-393, which vulkan-train-compiles-once reshapes.
  - Five unordered stories edit `verify.rs`.
- Out of my lane (acceptance): story:vulkan-step-counters tests `device allocations` against "a plan with a known buffer count" (vulkan-step-counters.md:42-43). Liveness later breaks the one-allocation-per-buffer link that test relies on, and does not list that test in its scope.
- Out of my lane (scope): the guide figures that profile-begin-stamp-after-barrier retakes in `website/docs/guides/inspect-a-run.md` go stale again once tiling, fuse and liveness change the dispatch list.

```findings
- file: .engineering/planning/story/matmul-shape-aware-tiling.md
  line: 34
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its outcome re-tiles every matmul site including k11, after story:profile-begin-stamp-after-barrier changes how --profile times every kernel, and story:weight-grad-matmul-structural's acceptance 1 (k11 at most 4.0 us from --profile) and its default-tile baseline read both, yet no blocks edge from this story to story:weight-grad-matmul-structural (nor depends_on the other way) records the order; adding blocks: story:weight-grad-matmul-structural here resolves it, since this story already depends on the stamp story"
```
