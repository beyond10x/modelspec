---
format: aep.planning-md/3
id: review-result:review-findings-acceptance-round-2
kind: review-result
status: active
title: Plan critic acceptance, review-finding stories, round 2
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

Round 1 status: 12 of my 14 findings are resolved. 2 still stand in a narrower form (checkpoint and observatory). 3 are new. Story line numbers are from the working copy, which has uncommitted revisions on top of HEAD 2b17433. Code line numbers are from HEAD.

- story:checkpoint-size-checks-never-wrap — partly resolved: the test that could not run on 64-bit is gone. But the new `declared-size-past-file` already passes at HEAD. The reader refuses both cases with E-CKPT-002 today (`crates/modelspec-runtime/src/train/checkpoint.rs:678-681`, `:741-745`), and tests already cover them (`crates/modelspec-runtime/tests/checkpoint_adversary.rs:191`, `:215`). That leaves the truncation outcome checkable only by reading the code. Fix: make `sizes-use-try-from` the gated check by naming the conversions it removes (`checkpoint.rs:685`, `:776`) as a grep that finds them before and not after, and mark `declared-size-past-file` as a regression guard — .engineering/planning/story/checkpoint-size-checks-never-wrap.md:33
- story:observatory-emit-copies-outside-hub-lock — partly resolved: a bound is named now. But `emit-never-waits` already passes at HEAD. A client that reads nothing blocks in `write_all` after `drop(frames)` (`crates/modelspec-observatory/src/server.rs:303-304`), so it never holds the lock that `emit` takes. Only the backlog `concat` runs under the lock (`server.rs:300`). Fix: replace the criterion with one that fails at HEAD, such as a test hook counting bytes copied while the hub lock is held, required to be 0 for a client catching up on a large backlog — .engineering/planning/story/observatory-emit-copies-outside-hub-lock.md:27
- story:fast-math-mode — new: `numerics-class-reported` prints `numerics class: tolerance`, which every M1 training run already prints. `numerics_class` puts any plan with a non-matmul kernel in tolerance (`crates/modelspec-runtime/src/vulkan/runner.rs:1068-1080`), so the output is the same before and after. Fix: test `numerics_class` with a bit-for-bit negotiation and a matmul-only plan, expecting `BitForBit` without `fast_math` and `Tolerance` with it — .engineering/planning/story/fast-math-mode.md:76
- story:mean-mse-adjoints-divide-by-n — new: the outcome says a test holds the code to the stated scale for `mean` and `mse`. `mean-adjoint-bits-n3` tests only `mean`, so the `mse` row's constant and its `(Y-T) * (G * c)` order (`crates/modelspec-core/src/ops.rs:270-277`) have no test. Fix: add an `mse` test over 3 elements that checks the folded `Scalar` bits equal `f32(2/3)` and the `Mul(d, Mul(G, c))` shape — .engineering/planning/story/mean-mse-adjoints-divide-by-n.md:40
- story:matmul-shape-aware-tiling — new; the text is the same as in round 1 and I missed it then. `padded-strides-measured` keeps padding "with a measured `--profile` delta" but gives no direction or threshold, so padding that slows the MNIST matmul also passes. Fix: say the padded stride is kept only when the MNIST matmul's `mean_us` falls by a named margin compared with the unpadded stride — .engineering/planning/story/matmul-shape-aware-tiling.md:47

**What I read:**
- All 28 ids in ids-r2.txt; 28 resolved through `aep plan artifact show` and all 28 bodies read.
- `aep plan artifact lifecycle story`, plus `git status` and `git diff` against 8ffb903.
- My round-1 report `.engineering/drafts/r1-acceptance.md` and `.engineering/drafts/r1-outcomes.md`.
- In the tree: checkpoint.rs:636-800 and both checkpoint adversary test files, server.rs:141-160 and :270-305, ops.rs:255-285, runner.rs:1068-1080, compile.rs (`--device-profile`), data.rs:80-100 and :360-380, DESIGN.md:4, :240-248, :301-305, :495 and :614.
- PR 30's ADR-008 text, read-only through `gh pr diff`.
- C2 to C10 in `review-0.5.0/REVIEW.md`.

**What I could not establish:**
- Whether the parser accepts the dimension 2305843009213693952 in `param-init-refuses-oversized-allocation`. The story's own fallback covers that case, and `read_tensor` does reach `zeros` before it reads the file (`data.rs:373`).
- I ran no code, so no timing or profile result here is measured.
- Out of my lane, so not in the verdict:
  - ADR-008 promises that two runs of one `fast_math` artifact give equal bits, and no `fast-math-mode` criterion checks that (scope critic).
  - `publish` builds `frame(event)` while it holds the hub lock (`server.rs:156`), which story:observatory-emit-copies-outside-hub-lock does not mention (design critic).

```findings
- file: .engineering/planning/story/checkpoint-size-checks-never-wrap.md
  line: 33
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`declared-size-past-file` already passes at HEAD (checkpoint.rs:678-681 and :741-745 refuse both cases with E-CKPT-002, tested at tests/checkpoint_adversary.rs:191 and :215), so the truncation outcome is checkable only by reading code; make `sizes-use-try-from` the gated check by naming the conversions it removes (checkpoint.rs:685, :776) as a grep that finds them before and not after, and mark `declared-size-past-file` as a regression guard"
- file: .engineering/planning/story/observatory-emit-copies-outside-hub-lock.md
  line: 27
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`emit-never-waits` already passes at HEAD, because a non-reading client blocks in `write_all` after `drop(frames)` (server.rs:303-304) and only the backlog concat runs under the lock (server.rs:300); replace it with a criterion that fails at HEAD, such as a test hook counting bytes copied under the hub lock, required to be 0 for a client catching up on a large backlog"
- file: .engineering/planning/story/fast-math-mode.md
  line: 76
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`numerics-class-reported` expects `numerics class: tolerance`, which every M1 training run already prints (runner.rs:1068-1080), so the output reads the same before and after; test `numerics_class` with a bit-for-bit negotiation and a matmul-only plan, expecting BitForBit without fast_math and Tolerance with it"
- file: .engineering/planning/story/mean-mse-adjoints-divide-by-n.md
  line: 40
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the outcome says a test holds the code to the stated scale for `mean` and `mse`, but `mean-adjoint-bits-n3` tests only `mean`, so the `mse` constant and its `(Y-T) * (G * c)` order (ops.rs:270-277) have no test; add an `mse` test over 3 elements that checks the folded Scalar bits equal f32(2/3) and the Mul(d, Mul(G, c)) shape"
- file: .engineering/planning/story/matmul-shape-aware-tiling.md
  line: 47
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`padded-strides-measured` keeps padding 'with a measured `--profile` delta' but gives no direction or threshold, so padding that slows the MNIST matmul also passes; say the padded stride is kept only when the MNIST matmul's `mean_us` falls by a named margin compared with the unpadded stride"
```
