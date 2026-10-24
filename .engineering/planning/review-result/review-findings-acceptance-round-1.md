---
format: aep.planning-md/3
id: review-result:review-findings-acceptance-round-1
kind: review-result
status: active
title: Plan critic acceptance, review-finding stories, round 1
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

- story:checkpoint-size-checks-never-wrap — `sizes-use-try-from` requires a size above `usize::MAX` to be refused by a "host test with a crafted header", but a `u64` above `usize::MAX` cannot exist on a 64-bit host, so the refusal can never be exercised there and the criterion reads the same before and after the work — .engineering/planning/story/checkpoint-size-checks-never-wrap.md:29
- story:cpu-prepared-training-step — `mnist-step-time` says the step time "is reported before and after" with no direction or threshold, so it passes when the step gets slower — .engineering/planning/story/cpu-prepared-training-step.md:36
- story:cpu-prepared-training-step — `intermediates-freed` ("a value is dropped after its last consumer") names no instrument (peak live-value count, drop hook) that shows when a value is dropped — .engineering/planning/story/cpu-prepared-training-step.md:33
- story:fast-math-mode — no criterion shows that a `fast_math = true` program actually emits an FMA, a cooperative-matrix kernel or a multi-workgroup loss sum (the outcome only says "may use"). `fast-math-hash` already holds, because the artifact hash covers `fast_math` today, and no acceptance names how `fast_math = true` is set — .engineering/planning/story/fast-math-mode.md:39
- story:fast-math-mode — `fast-math-throughput` only reports figures "against the deterministic figures" with no direction or threshold, so it passes with no gain — .engineering/planning/story/fast-math-mode.md:48
- story:matmul-shape-aware-tiling — `sweep-median` compares against 31067.1 µs measured at 0.6.0 (b2f9553), not at main 496e98e, so "falls" can be met by changes already landed since 0.6.0; the before state is not the one this work starts from — .engineering/planning/story/matmul-shape-aware-tiling.md:31
- story:matmul-shape-aware-tiling — `same-bits` asserts equality to the CPU oracle, but DESIGN.md § 5.4 asserts that only when negotiate gives `bit-for-bit`, and the RTX 3090 reports `shaderDenormPreserveFloat32` false, so on the benchmark device the check is not the defined one. It should say bit-identical to before on that device — DESIGN.md:307
- story:mean-mse-adjoints-divide-by-n — `mse-adjoint-bits` requires `(Y - T) * G * 2 / n` "in that order", but DESIGN.md § 4 states `dY = 2 * (Y-T) / numel(Y)` with no `G`, so the demanded rounding order is not the one the spec gives — DESIGN.md:246
- story:mean-mse-adjoints-divide-by-n — `vulkan-follows` points at "the class DESIGN.md § 5.4 assigns" without naming it, and the change moves the class: § 5.4 puts the `mse` gradient scale in the bit-for-bit class as a folded constant `mul`, while a runtime `G / n` is an `OpFDiv`, which § 5.4 puts in the tolerance class — DESIGN.md:304
- story:observatory-emit-copies-outside-hub-lock — `emit-never-waits` names no bound and describes a client that "does not read" yet "drains", and `frames-copied-after-unlock` is checkable only by reading the code, so neither gives a repeatable pass or fail — .engineering/planning/story/observatory-emit-copies-outside-hub-lock.md:26
- story:param-init-refuses-oversized-allocation — `param w: Tensor<f32,[65536,65536]>` asks for 16 GiB, which the host's RAM, swap and overcommit setting decide, so the outcome is not repeatable across hosts. The abort was inferred and never run (REVIEW.md C5 says "inferred"). The acceptance needs a shape no host can allocate — .engineering/planning/story/param-init-refuses-oversized-allocation.md:29
- story:schedule-index-tables — `ten-thousand-nodes` ("under 1 s" at one size) names no time before the work, and one point cannot tell quadratic from linear, so a program that is still quadratic can pass. The acceptance needs the before time or a size ratio, such as 10 000 against 20 000 nodes — .engineering/planning/story/schedule-index-tables.md:30
- story:vulkan-buffer-liveness-suballocation — `allocate-count` requires the `vkAllocateMemory` count to be "bounded by a constant independent of the number of values" but names no constant, so any count passes — .engineering/planning/story/vulkan-buffer-liveness-suballocation.md:39
- story:vulkan-scalar-operand-in-reduce-and-loss — `scalar-sum-trains-on-vulkan` and `scalar-mse-trains-on-vulkan` require the Vulkan per-step loss to "equal" or be "the CPU's loss", but § 5.4 puts the `mse` value after its division and any two-stage reduction in the tolerance class. Bit equality cannot be required and no tolerance is named — DESIGN.md:304

What I read: all 27 ids from ids.txt (27 read, none skipped), with `aep plan artifact show story:<id>` in full. I also read `aep plan artifact kinds` and `aep plan artifact lifecycle story`. I checked the tree with `rg` and `sed` against DESIGN.md:228, :246, :303-307, the artifact.rs and emit.rs `fast_math` fields, and checkpoint.rs:670-780. I also checked `verify_plan` and the `syntax.rs:1005` test name, the guide at website/docs/guides/inspect-a-run.md, and the review files.

What I could not establish:
- I did not run the code, so I could not confirm that `schedule-index-tables` fails its 1 s bound on current main, or that `cpu-matmul-output-skips-zero-fill` shows a memset in a profile.
- I could not tell whether the `fast_math` flag has a CLI or train-block switch. A grep finds only struct fields and `fast_math: false` defaults.
- Out of my lane, so not in the verdict:
  - `mean-mse-adjoints-divide-by-n` deliberately changes strict-mode gradient bits and moves the Vulkan SGD update out of the bit-for-bit class (design and scope critics).
  - `cpu-prepared-training-step` bundles three outcomes (scope critic).
  - `schedule.rs`, `verify.rs` and `emit.rs` are shared by 5 or more of these stories (parallel-safety critic).

```findings
- file: .engineering/planning/story/checkpoint-size-checks-never-wrap.md
  line: 29
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`sizes-use-try-from` requires a size above `usize::MAX` to be refused by a 'host test with a crafted header', but a `u64` above `usize::MAX` cannot exist on a 64-bit host, so the refusal can never be exercised there and the criterion reads the same before and after the work"
- file: .engineering/planning/story/cpu-prepared-training-step.md
  line: 36
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`mnist-step-time` says the step time 'is reported before and after' with no direction or threshold, so it passes when the step gets slower"
- file: .engineering/planning/story/cpu-prepared-training-step.md
  line: 33
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`intermediates-freed` ('a value is dropped after its last consumer') names no instrument (peak live-value count, drop hook) that shows when a value is dropped"
- file: .engineering/planning/story/fast-math-mode.md
  line: 39
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "no criterion shows that a `fast_math = true` program actually emits an FMA, a cooperative-matrix kernel or a multi-workgroup loss sum (the outcome only says 'may use'); `fast-math-hash` already holds because the artifact hash covers `fast_math` today, and no acceptance names how `fast_math = true` is set"
- file: .engineering/planning/story/fast-math-mode.md
  line: 48
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`fast-math-throughput` only reports figures 'against the deterministic figures' with no direction or threshold, so it passes with no gain"
- file: .engineering/planning/story/matmul-shape-aware-tiling.md
  line: 31
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`sweep-median` compares against 31067.1 us measured at 0.6.0 (b2f9553), not at main 496e98e, so 'falls' can be met by changes already landed since 0.6.0; the before state is not the one this work starts from"
- file: DESIGN.md
  line: 307
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`same-bits` in story:matmul-shape-aware-tiling asserts equality to the CPU oracle, but DESIGN.md 5.4 asserts that only when negotiate gives bit-for-bit, and the RTX 3090 reports `shaderDenormPreserveFloat32` false, so on the benchmark device the check is not the defined one; it should say bit-identical to before on that device"
- file: DESIGN.md
  line: 246
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`mse-adjoint-bits` in story:mean-mse-adjoints-divide-by-n requires `(Y - T) * G * 2 / n` 'in that order', but DESIGN.md 4 states `dY = 2 * (Y-T) / numel(Y)` with no `G`, so the demanded rounding order is not the one the spec gives"
- file: DESIGN.md
  line: 304
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`vulkan-follows` in story:mean-mse-adjoints-divide-by-n points at 'the class DESIGN.md 5.4 assigns' without naming it, and the change moves the class: 5.4 puts the mse gradient scale in the bit-for-bit class as a folded constant mul while a runtime `G / n` is an OpFDiv, which 5.4 puts in the tolerance class"
- file: .engineering/planning/story/observatory-emit-copies-outside-hub-lock.md
  line: 26
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "`emit-never-waits` names no bound and describes a client that 'does not read' yet 'drains', and `frames-copied-after-unlock` is checkable only by reading the code, so neither gives a repeatable pass or fail"
- file: .engineering/planning/story/param-init-refuses-oversized-allocation.md
  line: 29
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`param w: Tensor<f32,[65536,65536]>` asks for 16 GiB, which host RAM, swap and overcommit decide, so the outcome is not repeatable across hosts and the abort was inferred and never run; the acceptance needs a shape no host can allocate"
- file: .engineering/planning/story/schedule-index-tables.md
  line: 30
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`ten-thousand-nodes` ('under 1 s' at one size) names no time before the work, and one point cannot tell quadratic from linear, so a still-quadratic program can pass; it needs the before time or a size ratio such as 10 000 against 20 000 nodes"
- file: .engineering/planning/story/vulkan-buffer-liveness-suballocation.md
  line: 39
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`allocate-count` requires the `vkAllocateMemory` count to be 'bounded by a constant independent of the number of values' but names no constant, so any count passes"
- file: .engineering/planning/story/vulkan-scalar-operand-in-reduce-and-loss.md
  line: 31
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "`scalar-sum-trains-on-vulkan` and `scalar-mse-trains-on-vulkan` require the Vulkan per-step loss to 'equal' or be 'the CPU's loss', but DESIGN.md 5.4 puts the mse value after its division and any two-stage reduction in the tolerance class, so bit equality cannot be required and no tolerance is named"
```
