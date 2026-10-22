---
format: aep.planning-md/3
id: review-result:m1-scope-round-2
kind: review-result
status: active
title: Plan critic, scope lens, round 2, decomposition of epic:m1-vulkan-execution
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
approve

Round-1 findings of mine (all three resolved):
- `story:m1-inspect-device` — resolved: it now owns feature negotiation. `vulkan/negotiate.rs` is a pure function, Vulkan 1.2 and a compute queue are required, the logical device is created with exactly the set it returns, and E-TARGET-004 refuses a device below 1.2. Acceptance 3 and 4 check it. — `story:m1-inspect-device` "What it adds", bullet 4
- `story:m1-mlp-example` — resolved: Context says checkboxes 1-3 are already met by M0 and "does not claim them as new work". It names the GPU stories and `epic:pytorch-10x-mnist` as the reason, and carries `informed_by epic:pytorch-10x-mnist`. — `story:m1-mlp-example` Context, paragraphs 1-2
- `story:m1-compile-artifact` — resolved per R5: Context names the parent's `compile` verb, DESIGN § 7.1, § 8 and § 9 gate 8 as what it serves. The reader has a consumer: `compile` reads its own output back before exit 0. `emit_kernels` moved to `story:m1-vulkan-reduce-matmul`. — `story:m1-compile-artifact` Context and "What it adds", bullets 1 and 3

What I read: 12 artifacts, plus `.engineering/paused/revision-1-decisions.md`, the decomposer report, the DESIGN.md § 5.4 text in `design-10x.patch` and `aep plan artifact validate`, which is valid. Commands: `aep plan artifact show` on the epic, the ten stories and `review-result:m1-scope-round-1`, `aep plan artifact list --kind story`, `aep plan artifact graph`, and a grep for `UNMAPPED` in `spec/`. I did not run `aep plan artifact kinds` or `relations` this session. I did not read `m1-acceptance-round-1`, `m1-design-round-1` or `m1-parallel-safety-round-1`.

Promises: I extracted 26 from the epic body (outcome 1, runtime duties 5, kernel kinds 5, `.msckpt` 1, CLI verbs 4, checkboxes 9, spec-settling sentence 1). I traced all 26 to a story.
- Checkboxes 1-3 are claimed as already met by M0 in `story:m1-mlp-example`, which counts as covered under `epic:m0-semantic-core`.
- The `UNMAPPED` sentence is vacuous for M1. The only markers left in `spec/` are in `data.yaml` and `observatory.yaml`, and neither is an M1 type.

Could not establish, or out of my lane (none of these set the verdict):
- Overlap on timing output. `story:m1-vulkan-train` adds `--stats` and `story:m1-vulkan-profile` adds `--profile`, and both print the run counters and time per step on stderr. No acceptance item exercises `--stats`. I read this as one `RunStats` with two print paths and filed nothing.
- Overlap on `spirv-val`. `story:m1-vulkan-reduce-matmul` acceptance 2 and `story:m1-compile-artifact` acceptance 1 both check the five kernel kinds, one in memory and one on disk. I read it as one check on two surfaces and filed nothing.
- `story:m1-vulkan-reduce-matmul` takes the bit-for-bit class (RTE modes, bit-equality asserted on the operator run) into M1. The parent does not name it, but DESIGN § 5.4 0.9 does, and R3 decided it, so I did not reopen it.
- Design lane: § 5.4 0.9 item 1 has every kernel declare the float-control modes in deterministic mode. `story:m1-vulkan-elementwise` declares none and `story:m1-vulkan-reduce-matmul` declares them for two kernels only. That is a consistency question for the design critic.
- I did not check ordering edges or file overlaps; those are the parallel-safety lane.

```findings
[]
```
