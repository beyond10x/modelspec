---
format: aep.planning-md/3
id: review-result:m1-scope-round-1
kind: review-result
status: active
title: Plan critic, scope lens, round 1, decomposition of epic:m1-vulkan-execution
relations:
- reviews: epic:m1-vulkan-execution
revision: 1
---
needs-revision

- story:m1-inspect-device — "feature negotiation" is promised by the parent and no item claims it. The story only probes and prints features, and its sole refusal is E-TARGET-003 for no device or compute queue. Nothing states which features or Vulkan version the runtime requires, enables at logical-device creation, or refuses with a stated code. The kernel stories already build for `--target-env vulkan1.1`. This is the natural owner, as a "required features enabled, missing ones refused" item. — .engineering/planning/epic/m1-vulkan-execution.md:22 (draft: .engineering/drafts/m1-inspect-device.md:30)
- story:m1-mlp-example — nothing in the parent asks for this story's content. Checkboxes 1-3 are already met by M0, and the story's own Context says so. What it adds (the 784-64-10 MNIST-shape example, a golden file, a 36 s CPU run) comes from `epic:pytorch-10x-mnist`. The story claims checkboxes 1-3 as its outcome without saying they are pre-met. The body should say this and name the 10x lane as its reason, or move the example there. — .engineering/drafts/m1-mlp-example.md:9
- story:m1-compile-artifact — the validating reader with E-ARTIFACT-001 to 004 and seven tamper cases traces to DESIGN § 7.1 and § 9 gate 8, not to any parent sentence or checkbox. The parent promises only the `compile` verb. Nothing in M1 loads an artifact from disk: `train` "compiles the artifact in memory", and `infer` takes a source file. The body should name the parent sentence the reader serves, or the reader and its codes should be cut. — .engineering/drafts/m1-compile-artifact.md:26 (consumer check: .engineering/drafts/m1-vulkan-train.md:20)

**What I read.** 11 artifacts: the parent epic, the 10 drafts, `epic:pytorch-10x-mnist` and `epic:m0-semantic-core`. Commands: `aep plan artifact show` on each, `aep plan artifact graph`, and DESIGN.md § 1, 5, 6, 7, 8, 9, 11 and 12. I also read the decomposer report and grepped `spec/` for `UNMAPPED` (none in `artifact.yaml`, `checkpoint.yaml` or `device.yaml`).

**Promises.** I extracted 26 promises from the parent: the outcome, 5 runtime duties, 5 kernel kinds, `.msckpt`, 4 CLI verbs, 9 checkboxes and the spec-settling sentence. 25 trace to an item. One does not: feature negotiation.

I checked for earlier coverage. No story outside the set claims M1 items. The bert stories only reuse `GPU_ATOL` and `GPU_RTOL`, and `epic:pytorch-10x-mnist` lists M1 as its GPU lane.

**What I could not establish.** Out of my lane, and not counted toward the verdict:
- Parallel safety: `spirv/emit.rs` is edited by elementwise, reduce-matmul and compile-artifact, and `vulkan/` and `train.rs` are shared by several stories.
- Design: the artifact reader above has no consumer in M1.
- Design: `spirv/emit.rs` and `spirv/validate.rs` sit in `modelspec-runtime`, but `compile-artifact` has `modelspec-core`'s `artifact.rs` driving them.
- Acceptance and design: three bodies each assert "valid SPIR-V" for overlapping kernel sets (elementwise acc. 2, reduce-matmul acc. 2, compile-artifact acc. 1). I read this as a split by kind plus an end-to-end check, so I did not file it.
- Scope: whether M0's `check` accepts `device = vulkan(0)` in the new example.

```findings
- file: .engineering/planning/epic/m1-vulkan-execution.md
  line: 22
  category: scope
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "story:m1-inspect-device — \"feature negotiation\" is promised (\"Headless `ash` runtime: device discovery, feature negotiation, allocator, dispatch, sync\") and no drafted item claims it: inspect-device only probes and prints, its one refusal is E-TARGET-003 for no device or compute queue, and no story states the required features or Vulkan version or a diagnostic when they are missing"
- file: .engineering/drafts/m1-mlp-example.md
  line: 9
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:m1-mlp-example — the 784-64-10 example, golden IR and 1,800-step CPU run trace to epic:pytorch-10x-mnist, not to a parent sentence, and checkboxes 1-3 it claims are already met by M0 as its own Context says; the body must say so and name the 10x lane as the reason, or the example moves there"
- file: .engineering/drafts/m1-compile-artifact.md
  line: 26
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "story:m1-compile-artifact — the validating artifact reader, E-ARTIFACT-001 to 004 and seven tamper cases trace to DESIGN § 7.1 and § 9 gate 8, not to the parent's `compile` verb or any checkbox, and nothing in M1 loads an artifact from disk (train compiles in memory); the body must name the parent sentence it serves or drop the reader"
```
