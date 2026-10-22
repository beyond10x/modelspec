---
format: aep.planning-md/3
id: review-result:m1-acceptance-round-2
kind: review-result
status: active
title: Plan critic, acceptance lens, round 2, decomposition of epic:m1-vulkan-execution
relations:
- reviews: epic:m1-vulkan-execution
- reviews: story:m1-checkpoint-cpu
- reviews: story:m1-compile-artifact
- reviews: story:m1-inspect-device
- reviews: story:m1-kernel-plan
- reviews: story:m1-mlp-example
- reviews: story:m1-vulkan-checkpoint-infer
- reviews: story:m1-vulkan-elementwise
- reviews: story:m1-vulkan-profile
- reviews: story:m1-vulkan-reduce-matmul
- reviews: story:m1-vulkan-train
revision: 1
---
needs-revision
story:m1-vulkan-train — item 6 names one command, `cargo test -p modelspec-cli --test train_linear`, for a build "without the `vulkan` feature" and a build with it. `modelspec-cli` has `vulkan` on by default, so that command only builds the with-feature case. The E-TARGET-001 case, which backs the reworded status entry, has no stated way to be produced. — .engineering/planning/story/m1-vulkan-train.md:170 (default-on at .engineering/planning/story/m1-inspect-device.md:104)
story:m1-inspect-device — E-TARGET-003 and the no-feature E-TARGET-001 refusal are delivered in "What it adds", but no item produces either. Item 7 checks only that E-TARGET-003 is listed on the diagnostics page, so the codes can be listed and never emitted. — .engineering/planning/story/m1-inspect-device.md:108 and :188
story:m1-vulkan-profile — item 5 reads the same before and after the work. `task website` exits 0 without the guide, and a successful build does not show that the page holds the four inspection paths ("which runs ..."). The item names no observable for the page's content, such as the four command lines being present in it. — .engineering/planning/story/m1-vulkan-profile.md:106

Round-1 acceptance findings, all six resolved:
- **compile-artifact item 5 — resolved.** Every missing-loader check uses `MODELSPEC_VULKAN_LIBRARY` at a missing path. Item 6 states that every `compile --backend vulkan` in `task ci` is given `--device-profile` or that variable (m1-compile-artifact.md:109-125).
- **vulkan-train item 6 — resolved for the loader part.** It sets the variable. The build-feature gap above is a new finding.
- **inspect-device fixture order — resolved.** An "Order:" paragraph says the operator run produces the fixture and the coordinator commits it. It marks items 1, 3 and 7 as read after that commit, and the story as done only after the operator run (m1-inspect-device.md:154-157). The `depends_on` chain elementwise → inspect-device, reduce-matmul → elementwise and compile-artifact → reduce-matmul covers the other stories that read the fixture.
- **compile-artifact item 4 — resolved.** The nine cases are split into not re-sealed (002) and re-sealed (001 and 003), plus two that fail either way (004). Counted: 2 + 5 + 2 = 9.
- **checkpoint-cpu item 4 — resolved.** A counting `#[global_allocator]` records the largest single allocation per thread, bounded at file size + 4,096.
- **vulkan-profile item 1 — resolved.** The count is per dispatch site, with `calls` 1,800 for a step site and 1 for a forward site. Two sites sharing a module give two lines.

What I read: 10 of 10 story ids in full with `aep plan artifact show`, plus `epic:m1-vulkan-execution`, `review-result:m1-acceptance-round-1`, the decisions file `.engineering/paused/revision-1-decisions.md`, the rubric and the role file. I also checked the tree: `spec/domains/device.yaml` has 21 fields besides the identity, so 24 with the three new ones, which matches item 5. The `linear_regression` training golden has no `ReduceSum` or `ReduceMean`, so kernel-plan item 2 does not contradict item 3. The byte arithmetic in vulkan-train item 2 re-derives: 520,864 uploaded and 210,468 downloaded.

What I could not establish:
- **Operator-run bounds.** The 1e-4 whole-run L2 bound, the bit-for-bit class on the 3090 and the validation-layer counts need the device. Whether they are attainable is outside my lane.
- **Timestamp fallback.** vulkan-profile's `kernel timings unavailable` line is in "What it adds", and no item can exercise it on a CPU-only CI or on the 3090. This is an unease, not a finding.
- **Out of my lane, not counted in the verdict.** DESIGN.md edits to § 6 and § 6.3 on top of 0.9 are a design and parallel-safety matter. The reduce-matmul item 2 reference to the `mean` source defined in kernel-plan's `kernel_plan.rs` test is a parallel-safety matter too.

```findings
- file: .engineering/planning/story/m1-vulkan-train.md
  line: 170
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "item 6 names one command, cargo test -p modelspec-cli --test train_linear, for a build without the vulkan feature and a build with it, but the CLI vulkan feature is on by default, so the without-feature E-TARGET-001 case is not produced by the stated command (it needs --no-default-features or a separate item)"
- file: .engineering/planning/story/m1-inspect-device.md
  line: 108
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "E-TARGET-003 and the E-TARGET-001 refusal without the vulkan feature are delivered in What it adds, but no acceptance item produces either; item 7 only checks that E-TARGET-003 is listed on the diagnostics page"
- file: .engineering/planning/story/m1-vulkan-profile.md
  line: 106
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "item 5 'task website exits 0 with website/docs/guides/inspect-a-run.md, which runs ...' reads the same before and after the work and names no observable for the guide's content, so a build can pass with a guide that holds none of the four inspection paths"
```
