---
format: aep.planning-md/3
id: review-result:m0-acceptance-round-2
kind: review-result
status: active
title: Acceptance critic, M0 decomposition round 2
relations:
- reviews: story:m0-check-types-shapes
- reviews: story:m0-cpu-interpreter
- reviews: story:m0-cpu-sgd-training
- reviews: story:m0-lower-tensor-ir
- reviews: story:m0-parse-subset
- reviews: story:m0-reverse-mode-autodiff
- reviews: story:m0-tensor-ir-core
- reviews: story:m0-workspace-foundation
revision: 1
---
approve

I found nothing in the acceptance lane that would change what was drafted.

- **Round-1 finding (carried): fixed.** `story:m0-workspace-foundation` now has one outcome, `task ci` exits 0 on the committed tree, and it names the `no_vulkan` test with both cases (`.engineering/planning/story/m0-workspace-foundation.md:76-77`). The planted `ash v0.38.0` case is the automated proof that the guard can fail, and the story states that no manual mutation is part of the acceptance (lines 59-60).
- **Other seven stories.** Each acceptance is one sentence naming one runnable `cargo test` command with concrete values.
- **Two outcomes in one sentence.** `m0-parse-subset` and `m0-check-types-shapes` each join two checks with "and", but one test command asserts both. They do not split into independent done and not-done states, and I raised nothing on them in round 1 either.
- **Transitions.** Each CLI verb that was a "not implemented" stub with exit 1 now has an acceptance asserting exit 0 and exact output.
- **Numbers.** The values in the `m0-check-types-shapes` acceptance (`expected rhs dimension[0] = 32; got 64`) match the fixture it describes (lhs `[64, 32]`, rhs `[64, 128]`).

What I read: 8 of 8 story ids (`story:m0-*`), whole current bodies via `aep plan artifact show`, plus `review-result:m0-acceptance-round-1`, the acceptance-critic procedure and the rubric.

What I could not establish:
- I ran no `cargo` or `task` command, because the repository has no code, so I cannot confirm that the tolerances are achievable. These are the 1e-6 and 1e-3 gradient errors, the 1e-3 loss ratio and the 1e-2 weight bound.
- The `m0-reverse-mode-autodiff` acceptance cites "the tolerances above" instead of restating them, and its "relative error" does not say how a near-zero gradient is handled. I judged this an implementation detail and not a defect in the check.
- Out of my lane, and it did not set the verdict: the stories `m0-reverse-mode-autodiff`, `m0-lower-tensor-ir` and `m0-cpu-sgd-training` edit shared files (`ops.rs`, `ir/verify.rs`, `ir/training.rs`). That is for the parallel-safety critic.

```findings
[]
```
