---
format: aep.planning-md/3
id: review-result:review-findings-parallel-safety-round-2
kind: review-result
status: active
title: Plan critic parallel-safety, review-finding stories, round 2
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

- story:fuse-elementwise-chains-and-gradient-sums — the scope names the MNIST goldens but leaves out several files that pin the step this story fuses. `crates/modelspec-cli/tests/golden/linear_regression.kernel.ir` has the same `Sub`→`Mul` chain at :17-18 and is checked at `crates/modelspec-cli/tests/ir_kernel.rs:17`. Two tests pin the count: `ir_kernel.rs:96` asserts `(14, 8)` and `crates/modelspec-runtime/tests/vulkan_train.rs:213` asserts `14`. The "three fusions" text sits in DESIGN.md:164 and `spec/domains/artifact.yaml:227`, which `crates/modelspec-core/tests/adversary_fuse.rs:301,352` asserts. AGENTS.md:147-148 requires DESIGN.md to be edited with its revision line (DESIGN.md:4) bumped, so this story collides unnamed with story:mean-mse-adjoints-divide-by-n and story:param-init-refuses-oversized-allocation, and no edge orders them (the DESIGN.md and spec parts are inferred). Resolve by adding these files to the scope, or by an ordering edge that names DESIGN.md — .engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md:60
- story:vulkan-buffer-liveness-suballocation — the scope leaves out several surfaces that state or pin one buffer per value. The `buffer bN` list in `crates/modelspec-cli/tests/golden/linear_regression.kernel.ir:1-9` is checked at `ir_kernel.rs:17`. `spec/domains/artifact.yaml:227` says "each stored value has its own buffer", and `adversary_fuse.rs:297-358` asserts it; a change under `spec/` also regenerates `crates/modelspec-observatory/schema/` (AGENTS.md:68). DESIGN.md:164 says "one buffer and one view per value", and AGENTS.md:147-148 requires a revision bump with that edit. So this story collides unnamed with story:mean-mse-adjoints-divide-by-n and story:param-init-refuses-oversized-allocation on DESIGN.md, and no edge orders them (the spec and DESIGN.md parts are inferred). Resolve by adding the golden, the spec, the adversary test and DESIGN.md to the scope, or by an ordering edge that names DESIGN.md — .engineering/planning/story/vulkan-buffer-liveness-suballocation.md:63
- story:matmul-shape-aware-tiling — DESIGN.md:164 ("128 for every shape today") and DESIGN.md:275 state the tile rule this story changes, but the scope does not list DESIGN.md. Under AGENTS.md:147-148 the story then collides unnamed on DESIGN.md:4 with story:mean-mse-adjoints-divide-by-n, story:param-init-refuses-oversized-allocation and story:vulkan-step-counters, and neither an edge nor a shared scope orders it with any of them (inferred). Resolve by adding DESIGN.md to the scope — .engineering/planning/story/matmul-shape-aware-tiling.md:59
- story:vulkan-step-counters — the story adds three counters, but the scope leaves out the files that pin `RunStats` and the counter output. The struct literals list every field: `crates/modelspec-runtime/tests/vulkan_profile.rs:52` and `crates/modelspec-runtime/tests/adversary_profile_throughput.rs:26`. The counter lines are pinned byte for byte at `vulkan_profile.rs:188-193` and `:212-217`. The guide quotes the counter block at `website/docs/guides/inspect-a-run.md:129-136`. The `--stats`/`--profile` help text is at `crates/modelspec-cli/src/train.rs:191-199` and generates `website/docs/reference/cli.md:147-148`. The guide and `train.rs` entries would collide unnamed with story:matmul-shape-aware-tiling, story:serve-ends-without-signal-handler and story:vulkan-train-compiles-once, and none of them is ordered with this story (the guide and help entries are inferred). Resolve by adding these files to the scope — .engineering/planning/story/vulkan-step-counters.md:52
- story:fast-math-mode — `--fast-math` on `compile` and `train` changes the generated CLI reference `website/docs/reference/cli.md`. That page is built from the clap definition (`crates/modelspec-docs/src/main.rs:5`, `:212`) and checked for drift by `task check`, and the scope does not list it. Resolve by adding the page to the scope — .engineering/planning/story/fast-math-mode.md:83
- story:param-init-refuses-oversized-allocation — the story adds a named `E-` code (DESIGN.md § 8.1). `website/docs/reference/diagnostics.md` is generated from the code constants of each stage (`crates/modelspec-docs/src/diagnostics.rs:1-7`), and `train/data.rs` is one of those stages (`diagnostics.rs:52-53`). The scope does not list that page (inferred: the body does not say which file declares the code). Resolve by adding the page to the scope — .engineering/planning/story/param-init-refuses-oversized-allocation.md:59

**Round-1 findings:**

| Round-1 finding | Now |
|---|---|
| mean-mse surface understated | Resolved: scope is DESIGN.md plus a new test file, no change to the IR (mean-mse-adjoints-divide-by-n.md:45-52) |
| fuse goldens unnamed | Partly resolved: the MNIST goldens and the guide are named (fuse...md:62-67). The rest is the new finding above |
| tiling goldens unnamed | Resolved (matmul-shape-aware-tiling.md:59-64). The DESIGN.md finding above is new |
| liveness goldens unnamed | Resolved for MNIST and the guide (liveness...md:65-70). The finding above is new |
| profile-begin `website/docs` directory entry | Resolved: the scope names the page, and the acceptance names it at profile-begin-stamp-after-barrier.md:31 |
| fast-math / loss-slots order | Resolved: `depends_on` at fast-math-mode.md:9, and the Why section at :47-48 |

**What I read:** all 28 stories (`aep plan artifact show` on each id), `aep plan artifact waves --status draft --format json` (no cycles; none of the 28 is in `unassessed`) and `aep plan artifact relations`, against the tree at 2b17433 with the stories as they are in the working tree.
- Placement: all 28 have a cited surface. 20 have every entry cited, 8 have at least one inferred entry, and 0 could not be placed.
- Acceptance reads: 6 trace to a producer in the set, and all 6 have a `depends_on` edge:
  - fuse→counters
  - liveness→counters
  - liveness→fuse
  - tiling→profile-begin
  - fast-math→tiling
  - fast-math→loss-slots

  fast-math's `--profile` read reaches profile-begin through tiling.
- Collisions between declared scopes are not findings, because `waves` already separates them.

**What I could not establish:**
- Whether story:vulkan-loss-slots-written-by-kernel changes printed plan text (`loss b7`, the k2 signature) or DESIGN.md. The runner could rebind the loss view each step without changing the plan.
- Whether shape-aware tiling changes the `tile` lines of `linear_regression.kernel.ir` (:15, :20), whose shapes are already small.
- Whether story:checkpoint-reader-refuses-out-of-order-records adds an `E-CKPT` code or reuses E-CKPT-001. A new code would regenerate `diagnostics.md`.
- Where story:profile-begin-stamp-after-barrier and story:vulkan-pipeline-cache put their host tests. If either lands in `vulkan_profile.rs`, it collides with story:vulkan-step-counters there.
- The unordered DESIGN.md pairs above are in separate waves today only because of other chains, not because of a recorded order.

```findings
- file: .engineering/planning/story/fuse-elementwise-chains-and-gradient-sums.md
  line: 60
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope omits crates/modelspec-cli/tests/golden/linear_regression.kernel.ir (same Sub->Mul chain at :17-18, checked at ir_kernel.rs:17), the count pins ir_kernel.rs:96 (14, 8) and modelspec-runtime/tests/vulkan_train.rs:213 (14), and the three-fusions text in DESIGN.md:164 and spec/domains/artifact.yaml:227 (asserted by adversary_fuse.rs:301,352); the DESIGN.md edit with its revision bump (AGENTS.md:147-148) collides unnamed and unordered with story:mean-mse-adjoints-divide-by-n and story:param-init-refuses-oversized-allocation (DESIGN.md and spec parts inferred)"
- file: .engineering/planning/story/vulkan-buffer-liveness-suballocation.md
  line: 63
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope omits the buffer list of crates/modelspec-cli/tests/golden/linear_regression.kernel.ir:1-9 (ir_kernel.rs:17), spec/domains/artifact.yaml:227 'each stored value has its own buffer' asserted by adversary_fuse.rs:297-358 (a spec change also regenerates crates/modelspec-observatory/schema/, AGENTS.md:68), and DESIGN.md:164 'one buffer and one view per value', whose edit and revision bump (AGENTS.md:147-148) collide unnamed and unordered with story:mean-mse-adjoints-divide-by-n and story:param-init-refuses-oversized-allocation (spec and DESIGN.md parts inferred)"
- file: .engineering/planning/story/matmul-shape-aware-tiling.md
  line: 59
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "DESIGN.md:164 ('128 for every shape today') and DESIGN.md:275 state the tile rule this story changes, but scope omits DESIGN.md, so under AGENTS.md:147-148 it collides unnamed on the revision line DESIGN.md:4 with story:mean-mse-adjoints-divide-by-n, story:param-init-refuses-oversized-allocation and story:vulkan-step-counters, none ordered with it (inferred)"
- file: .engineering/planning/story/vulkan-step-counters.md
  line: 52
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope omits the exhaustive RunStats literals (modelspec-runtime/tests/vulkan_profile.rs:52, adversary_profile_throughput.rs:26), the byte-pinned profile counter lines (vulkan_profile.rs:188-193, :212-217), the guide's quoted counter block (website/docs/guides/inspect-a-run.md:129-136) and the --stats/--profile help (crates/modelspec-cli/src/train.rs:191-199, generating website/docs/reference/cli.md:147-148); the guide and train.rs entries collide unnamed with story:matmul-shape-aware-tiling, story:serve-ends-without-signal-handler and story:vulkan-train-compiles-once, none ordered with it (guide and help entries inferred)"
- file: .engineering/planning/story/fast-math-mode.md
  line: 83
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the --fast-math flag on compile and train changes website/docs/reference/cli.md, generated from the clap definition (crates/modelspec-docs/src/main.rs:5, :212) and drift-checked by task check, and the scope omits that page"
- file: .engineering/planning/story/param-init-refuses-oversized-allocation.md
  line: 59
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the new named E- code changes website/docs/reference/diagnostics.md, generated from each stage's code constants (crates/modelspec-docs/src/diagnostics.rs:1-7, train/data.rs a stage at :52-53), and the scope omits that page (inferred: the body does not say which file declares the code)"
```
