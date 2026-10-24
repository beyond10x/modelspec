---
format: aep.planning-md/3
id: review-result:review-findings-design-round-1
kind: review-result
status: active
title: Plan critic design, review-finding stories, round 1
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

story:mean-mse-adjoints-divide-by-n — the outcome `G / n` needs a division op, but `TensorOp` has only `Add`, `Sub` and `Mul`, the scope names only `autodiff.rs` and `ops.rs`, and no story in the set adds a `Div` through the interpreter, scheduler, emitter, verifier and DESIGN.md. Add that op as its own story and give this one a `depends_on` edge to it, or state how `G / n` is formed from existing ops — crates/modelspec-core/src/ir/tensor.rs:130-132; .engineering/planning/story/mean-mse-adjoints-divide-by-n.md:23,31

story:weight-grad-matmul-structural — its acceptance 2 pins the 1,800-step MNIST stdout sha `b9bb028a…6904`, and story:mean-mse-adjoints-divide-by-n changes those gradient bits (mse over `[100,10]`, n = 1000, not a power of two), yet no `depends_on` edge runs from it to that story — .engineering/planning/story/weight-grad-matmul-structural.md:34; examples/mnist_mlp.mspec:12-18; crates/modelspec-core/src/ops.rs:325-335

story:weight-grad-matmul-structural — its acceptance 1 (k11 at most 4.0 µs from `--profile`) and its "default tile" baseline read figures that story:matmul-shape-aware-tiling re-tiles for every MNIST matmul site and story:profile-begin-stamp-after-barrier re-measures (the "barrier chain" it cites is what the TOP_OF_PIPE stamp includes). A `depends_on` edge to story:matmul-shape-aware-tiling closes both, because that story already depends on the stamp story — .engineering/planning/story/weight-grad-matmul-structural.md:19,25,33; .engineering/planning/story/matmul-shape-aware-tiling.md:21-24

story:fuse-elementwise-chains-and-gradient-sums — its acceptance `barriers-per-step` reads a counter no item creates: `RunStats` has `dispatches_per_step` and no barrier count, the `--profile` report prints none, and the scope names neither file. Add the counter to this story's scope, or draft one counters story that this one and story:vulkan-buffer-liveness-suballocation both `depends_on` — .engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md:35; crates/modelspec-runtime/src/vulkan/runner.rs:75-95; crates/modelspec-runtime/src/vulkan/profile.rs:257-267

story:vulkan-buffer-liveness-suballocation — its acceptances `peak-device-bytes` and `allocate-count` read counters no item creates (neither is in `RunStats` or the profile report, and `profile.rs` is not in the scope). Its own `depends_on` edge to story:fuse-elementwise-chains-and-gradient-sums does not say who adds them, so it must put them in scope or depend on the shared counters story — .engineering/planning/story/vulkan-buffer-liveness-suballocation.md:37,39; crates/modelspec-runtime/src/vulkan/runner.rs:75-95

story:fast-math-mode — its multi-workgroup loss sum and story:vulkan-loss-slots-written-by-kernel's "loss kernel writes slot k" are two halves of one loss kernel, split by workgroup count and recorded by no edge. Add `depends_on` to story:vulkan-loss-slots-written-by-kernel, or move the parallel loss sum into its own story that depends on it (the FMA and cooperative-matrix parts are emit-only and stay) — .engineering/planning/story/fast-math-mode.md:39-40; .engineering/planning/story/vulkan-loss-slots-written-by-kernel.md:25,43

**What I read.** 27 drafted stories in full, plus 6 existing ones (weight-grad-matmul-structural, input-readers-bound-before-allocating, manifest-arrays-bounded-on-untrusted-artifact, vulkan-pool-created-only-from-room, vulkan-validation-messages-fail-the-run, observatory-ui-keeps-up-with-fast-runs) and TRIAGE.md. Commands: `aep plan artifact show` (each story), `relations`, `list --format json`, `validate`, and `graph` (output truncated, so I walked the edges from the JSON).

- **Edges.** I walked 7 edges declared by the set and all roughly 110 `depends_on` and `blocks` edges store-wide, so the walk went outside the set. There is no cycle.
- **Chain.** The only chains are profile-stamp → tiling → fast-math and fuse → liveness. Nothing serialises the set.
- **Acceptance reads.** I traced 9 acceptance reads to a producer or its absence. Two have a producer in the set with an edge (the tiling and fast-math `--profile` figures read the stamp story). Two have a producer in the set and no edge (the pinned sha, and k11 timing against tiling). Three have no producer in the set (barrier count, peak bytes, allocation count). The Div op is a separate gap.
- **Validate.** `aep plan artifact validate` reports valid.

**What I could not establish.**
- Whether the four readers in story:input-readers-bound-before-allocating are the `data.rs` readers that story:param-init-refuses-oversized-allocation edits. The story keeps its details in an external review. That is an overlap question for the parallel-safety and scope critics.
- Out of my lane (parallel safety): these stories share files and are unordered.
  - `schedule.rs` and `emit.rs`: fuse, vulkan-scalar-operand, loss-slots, liveness, fast-math.
  - `verify.rs`: verifier-counts-every-parameter-write, vulkan-scalar-operand, schedule-index-tables, cpu-prepared-training-step.
  - The streaming loop at `observatory/src/server.rs:283-305`: O2 and O3 edit adjacent lines.
  - `parser.rs`: three stories.
  - `checkpoint.rs`: two stories.
- Out of my lane (scope and acceptance):
  - story:param-init-refuses-oversized-allocation says the allocation has no fallible path, but `zeros` at `data.rs:456-470` already returns `TooManyElements` on overflow.
  - story:cpu-prepared-training-step's `mnist-step-time` and the profile-figure acceptances carry no threshold.
- I found no merge worth making. cpu-prepared-training-step is one prepare-and-run abstraction. The ADR gate on story:fast-math-mode is an operator decision outside the set.

```findings
- file: .engineering/planning/story/mean-mse-adjoints-divide-by-n.md
  line: 23
  category: design
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "the outcome G / n needs a division op, but TensorOp has only Add, Sub and Mul (crates/modelspec-core/src/ir/tensor.rs:130-132), the scope names only autodiff.rs and ops.rs, and no story in the set adds a Div through interpreter, scheduler, emitter and verifier; add that op as its own story with a depends_on edge to it, or state how G / n is formed from existing ops"
- file: .engineering/planning/story/weight-grad-matmul-structural.md
  line: 34
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its acceptance 2 pins the 1,800-step MNIST stdout sha, and story:mean-mse-adjoints-divide-by-n changes those gradient bits (mse over [100,10], n = 1000), yet no depends_on edge runs from it to that story"
- file: .engineering/planning/story/weight-grad-matmul-structural.md
  line: 33
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its acceptance 1 and its default-tile baseline read k11 figures that story:matmul-shape-aware-tiling re-tiles and story:profile-begin-stamp-after-barrier re-measures, and no depends_on edge to story:matmul-shape-aware-tiling (which already depends on the stamp story) records it"
- file: .engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md
  line: 35
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its acceptance barriers-per-step reads a counter no item creates (RunStats has no barrier count, the --profile report prints none, the scope names neither file); add the counter to its scope or draft one counters story that it and story:vulkan-buffer-liveness-suballocation depend on"
- file: .engineering/planning/story/vulkan-buffer-liveness-suballocation.md
  line: 37
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its acceptances peak-device-bytes and allocate-count read counters no item creates (not in RunStats or the profile report, profile.rs not in scope); put them in its scope or depend on the shared counters story"
- file: .engineering/planning/story/fast-math-mode.md
  line: 39
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its multi-workgroup loss sum and story:vulkan-loss-slots-written-by-kernel's loss-kernel-writes-slot are two halves of one loss kernel with no edge between them; add depends_on to story:vulkan-loss-slots-written-by-kernel, or split the parallel loss sum into its own story that depends on it"
```
