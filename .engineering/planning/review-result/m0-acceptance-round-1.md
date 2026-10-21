---
format: aep.planning-md/3
id: review-result:m0-acceptance-round-1
kind: review-result
status: active
title: Acceptance critic, M0 decomposition round 1
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
needs-revision

story:m0-workspace-foundation — the acceptance joins two independent outcomes with "and" (`task ci` exits 0, and the `core-no-vulkan` guard fails on a Vulkan dependency), and the second is a manual mutation that no named test or command asserts, so the guard can be broken while the artifact still reads as done — .engineering/planning/story/m0-workspace-foundation.md:69

What I read: 8 of 8 story ids (`story:m0-*`), whole bodies via `aep plan artifact show` for each. I also ran `aep plan artifact kinds` and `aep plan artifact lifecycle story`, and read `epic:m0-semantic-core`, `examples/linear_regression.mspec` and DESIGN.md §§ 8.1 and 9. The other seven acceptances each name one runnable `cargo test` command with concrete values, so I raised nothing on them. Their numbers match the example file and the DESIGN.md § 8.1 text.

What I could not establish:
- I did not run any `cargo` or `task` command, because the repository has no code yet. I could not check that the acceptances' tolerances and thresholds (1e-6 and 1e-3 gradient error, loss ratio 1e-3, weights within 1e-2) can be met. The `m0-cpu-sgd-training` context gives its own margin estimate (about 2e-4 parameter error), which I did not verify.
- `story:m0-cpu-sgd-training` (`.engineering/planning/story/m0-cpu-sgd-training.md:84`) asserts closeness to "the teacher weights" without saying how the test obtains them. The Lands-in section implies a public runtime function. I judged that an implementation detail, so it is not a finding.
- Out of my lane, and it did not set the verdict: `m0-reverse-mode-autodiff` edits `ops.rs`, `ir/verify.rs` and `ir/training.rs`, which `m0-lower-tensor-ir` and `m0-cpu-sgd-training` also touch. That is for the parallel-safety critic.

```findings
- file: .engineering/planning/story/m0-workspace-foundation.md
  line: 69
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the acceptance joins two independent outcomes with 'and' (`task ci` exits 0, and the `core-no-vulkan` guard fails on a Vulkan dependency), and the second is a manual mutation that no named test or command asserts, so the guard can be broken while the artifact still reads as done"
```
