---
format: aep.planning-md/3
id: review-result:review-findings-parallel-safety-round-1
kind: review-result
status: active
title: Plan critic parallel-safety, review-finding stories, round 1
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

- story:mean-mse-adjoints-divide-by-n — its scope (`ops.rs`, `autodiff.rs`) cannot deliver `G / n`: `TensorOp` has no division op, so landing the formula needs a new op across tensor/print/verify/interpreter/emit, and the `Scalar 0.002` it removes sits in the committed `mnist_mlp.training.ir` and `mnist_mlp.kernel.ir` goldens, so the surface is understated and unassessable (inferred) — .engineering/planning/story/mean-mse-adjoints-divide-by-n.md:35-38, crates/modelspec-core/src/ir/tensor.rs:128-142, crates/modelspec-cli/tests/golden/mnist_mlp.training.ir:14
- story:fuse-elementwise-chains-and-gradient-sums — it changes the dispatch list that `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir` and the quoted kernel lines of `website/docs/guides/inspect-a-run.md` pin byte for byte, and its scope names neither file, so it collides unnamed with story:matmul-shape-aware-tiling, story:vulkan-buffer-liveness-suballocation and story:mean-mse-adjoints-divide-by-n on the goldens (inferred) — .engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md:40-44, crates/modelspec-cli/tests/ir_kernel.rs:36, crates/modelspec-cli/tests/adversary_guide_kernel_lines.rs:1-4
- story:matmul-shape-aware-tiling — a shape-dependent tile changes the `tile … cols rows depth` text and the kernel ids printed in the kernel goldens and the guide's kernel lines (kernel ids are derived from the tile per story:kernel-signature-covers-matmul-tile), and its scope lists neither, so it collides unnamed with story:fuse-elementwise-chains-and-gradient-sums and story:vulkan-buffer-liveness-suballocation on them (inferred) — .engineering/planning/story/matmul-shape-aware-tiling.md:41-44, crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir:25
- story:vulkan-buffer-liveness-suballocation — reusing storage changes the `buffer bN` list and the view bindings in the kernel goldens, which its scope does not list, so it collides unnamed with story:fuse-elementwise-chains-and-gradient-sums and story:matmul-shape-aware-tiling on those files (inferred) — .engineering/planning/story/vulkan-buffer-liveness-suballocation.md:47-52, crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir:1
- story:profile-begin-stamp-after-barrier — scope `website/docs` is a whole directory marked inferred, the acceptance's "the guide" resolves to `website/docs/guides/inspect-a-run.md` (the only page with the RTX 3090 profile figures), and the directory entry hides its overlap with story:crates-page-names-current-release (`website/docs/reference/crates.md`) from `waves` collisions; name the page (inferred) — .engineering/planning/story/profile-begin-stamp-after-barrier.md:31, .engineering/planning/story/profile-begin-stamp-after-barrier.md:39
- story:fast-math-mode — its multi-workgroup loss sum rewrites the loss kernel that story:vulkan-loss-slots-written-by-kernel makes write the step's loss slot (schedule.rs and emit.rs, the loss-slots entries inferred), loss-slots' out-of-scope line names this story but this body and its edges name neither it nor an ordering, and the only recorded link is that one-sided sentence — .engineering/planning/story/fast-math-mode.md:54-59, .engineering/planning/story/vulkan-loss-slots-written-by-kernel.md:43

**What I read:** 27 stories (`aep plan artifact show` on each id in ids.txt) plus `aep plan artifact waves --status draft --format json` (collisions, cycles, unassessed), against the tree at 8ffb903. Scope placement: all 27 have a cited primary surface (17 with every entry cited, 10 with at least one inferred entry), and 0 are unplaceable. Of the acceptance reads I traced to a producer in the set, 4 have a producer and 3 of those are recorded by `depends_on` (fast-math→tiling, tiling→profile-begin, liveness→fuse). The fourth is the loss-slots/fast-math finding above.

Collisions between declared scopes are not findings, because the typed scopes already state them and `waves` splits them into separate waves. By file, these are the groups:

| File | Stories | Edges |
|---|---|---|
| `schedule.rs` | fast-math, fuse, liveness, loss-slots, scalar-operand, schedule-index | one edge (liveness→fuse) |
| `emit.rs` | fast-math, fuse, tiling, loss-slots, scalar-operand | fast-math→tiling |
| `verify.rs` | cpu-prepared, verifier-counts, liveness, scalar-operand, schedule-index | none |
| vulkan `runner.rs` | profile-begin, liveness, loss-slots, pipeline-cache | none |
| `server.rs` | the 3 observatory stories | none |
| `parser.rs` | float-literals, span, unclosed-train | none |
| `checkpoint.rs` | the 2 checkpoint stories | none |
| `train/runner.rs` | cpu-finite-check, dense-step | none |
| `train/data.rs` | dense-step, param-init | none |
| cli `train.rs` | serve-ends, train-compiles-once | none |

**What I could not establish:**
- Whether story:mean-mse-adjoints-divide-by-n really needs a new division op. I read none of the other stories' emit/schedule code, so this is inferred from `TensorOp` having no `Div`.
- Whether story:vulkan-loss-slots-written-by-kernel changes any printed golden line. The plan print shows the loss binding but not the per-step slot, so I left it out.
- Several pairs rest only on inferred entries (loss-slots against schedule.rs/emit.rs, scalar-operand and liveness against verify.rs, fuse against mean-mse on `autodiff.rs`). If those entries are wrong, the ordering `waves` computed disappears.
- Out of my lane: story:mean-mse-adjoints-divide-by-n's `G / n` with no division op is a design and acceptance question. The draft set also collides with the bert-* stories and has 17 `unassessed` entries in `waves`, but those are outside the set I was given.

```findings
- file: .engineering/planning/story/mean-mse-adjoints-divide-by-n.md
  line: 35
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "scope lists ops.rs and autodiff.rs (autodiff.rs inferred) but TensorOp has no division op, so G / n needs a new op across tensor/print/verify/interpreter/emit and rewrites the Scalar 0.002 lines in the committed training and kernel goldens; the surface is understated and unassessable (inferred)"
- file: .engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md
  line: 40
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope omits crates/modelspec-cli/tests/golden/*.kernel.ir and website/docs/guides/inspect-a-run.md, both pinned byte for byte to the kernel plan this story changes, so it collides unnamed with story:matmul-shape-aware-tiling, story:vulkan-buffer-liveness-suballocation and story:mean-mse-adjoints-divide-by-n (inferred)"
- file: .engineering/planning/story/matmul-shape-aware-tiling.md
  line: 41
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope omits the kernel goldens and the guide's quoted kernel lines that print the tile and kernel ids this story changes, so it collides unnamed with story:fuse-elementwise-chains-and-gradient-sums and story:vulkan-buffer-liveness-suballocation (inferred)"
- file: .engineering/planning/story/vulkan-buffer-liveness-suballocation.md
  line: 47
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope omits the kernel goldens whose buffer list and views change when storage is shared, so it collides unnamed with story:fuse-elementwise-chains-and-gradient-sums and story:matmul-shape-aware-tiling (inferred)"
- file: .engineering/planning/story/profile-begin-stamp-after-barrier.md
  line: 39
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope website/docs is a whole directory marked inferred; the acceptance's guide is website/docs/guides/inspect-a-run.md, and the directory entry hides the overlap with story:crates-page-names-current-release from the waves collision check (inferred)"
- file: .engineering/planning/story/fast-math-mode.md
  line: 54
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the multi-workgroup loss sum edits the loss kernel in schedule.rs and emit.rs that story:vulkan-loss-slots-written-by-kernel changes (its entries there inferred); only loss-slots' out-of-scope line names this story, and no depends_on edge or mention here records the order"
```
