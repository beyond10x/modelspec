---
format: aep.planning-md/3
id: review-result:m1-design-round-2
kind: review-result
status: active
title: Plan critic, design lens, round 2, decomposition of epic:m1-vulkan-execution
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

- story:m1-vulkan-reduce-matmul — The same predicate, "device reports all three float-control values", is decided in three places. inspect-device `negotiate` reports a class, this story's emitter and its `class:` line re-read the three booleans, and train's `RunStats` names a class. Nothing says the emitter and the printed class come from the one `negotiate` result, so the module modes and the reported class can disagree. The body should say both use `negotiate`'s class — `.engineering/planning/story/m1-vulkan-reduce-matmul.md:163`, `.engineering/planning/story/m1-inspect-device.md:173`
- story:m1-compile-artifact — `compile --backend vulkan` accepts a DeviceProfile and emits `vulkan1.2` modules keyed to it, but the body never applies the E-TARGET-004 version gate that inspect-device puts in `negotiate`. A profile below 1.2, from `--device-profile` or from the probe, compiles to modules its device cannot run. The body should say `compile` runs `negotiate` and exits 1 with E-TARGET-004. Acceptance 5 needs a 1.1.0 copy of the fixture. — `.engineering/planning/story/m1-compile-artifact.md:60`, `.engineering/planning/story/m1-compile-artifact.md:118`, `.engineering/planning/story/m1-inspect-device.md:103`
- story:m1-kernel-plan — The step plan now contains an `Allocate` step for the partials buffer, and vulkan-train records the step plan once per step. No body says whether `Allocate` runs once per run or once per recording, so the plan's replay contract is split between the plan owner, reduce-matmul's `Allocate` handling and train. The body, and DESIGN § 7.1, should state that a step plan's `Allocate` steps are hoisted to run setup and that the verifier checks this. — `.engineering/planning/story/m1-kernel-plan.md:94`, `.engineering/planning/story/m1-vulkan-train.md:85`, `.engineering/planning/story/m1-vulkan-reduce-matmul.md:99`

Round-1 findings of mine, nine in all, every one resolved:
- m1-kernel-plan, two-stage reduce: resolved, since the four-step lowering, verifier rules and acceptance 3 and 4 are present (`m1-kernel-plan.md:94`, `:162`).
- m1-vulkan-reduce-matmul, spec edit: resolved, since `artifact.yaml` is in scope and `ess specify validate` is in acceptance 1.
- m1-vulkan-reduce-matmul, rule wording and § 5.4 class: resolved, since it now says "an order fixed by shape alone", with a compile-time workgroup size and the CPU order as one admissible order. M1 emits the three modes and asserts bit-equality (acceptance 6).
- m1-vulkan-elementwise, numerics split: resolved, since `r` there covers elementwise, broadcast and SGD only and reduce-matmul extends gate 5.
- m1-checkpoint-cpu, DESIGN ownership: resolved per decision R2. The one-line append-only note is in each Scope, and no edge is added for these surfaces alone.
- m1-vulkan-train, § 6 on top of 0.9: resolved, since it says the edit applies on 0.9 and § 6.3 keeps only the M2 items 1, 2, 3, 6 and 7.
- m1-inspect-device, three float-control fields: resolved, since they are in `device.yaml`, the probe, acceptance 4 to 6 and the fixture.
- m1-kernel-plan, `ParameterSpec.views`: resolved. One view is in-place, two is ping-pong, and the verifier accepts one or two.
- m1-compile-artifact, bundling: resolved. `emit_kernels` moved to reduce-matmul, train no longer depends on compile-artifact, and compile-artifact's only consumer is profile.

Decision 7 (Vulkan 1.2 required, float controls recorded, E-TARGET-004): I judge it sound. Float controls and SPIR-V 1.5 are core in Vulkan 1.2, and the `spirv-val` target `vulkan1.2` agrees across the stories. Recording the three fields keeps the § 5.4 tolerance fallback. Only the two seams above remain, the class predicate and the compile path. `E-TARGET-004` has no other user in the tree.

Edges and no-cycle check: I found no needs-first cycle, and the set is not a queue (four stories can start in parallel, and train runs beside compile-artifact). The ordering edges for shared files, such as `elementwise→mlp-example` and `compile-artifact→checkpoint-cpu`, each state their file in the body.

What I read: 11 artifacts (the epic and 10 stories) plus `review-result:m1-design-round-1` and `.engineering/paused/revision-1-decisions.md`, via `aep plan artifact show`, `relations`, `graph`, `waves` and `validate` (valid, 88 artifacts). I also read the patch and its notes, `status.rs`, `examples/linear_regression.mspec` and its golden training IR. I walked about 45 edges, including those to `epic:m0-semantic-core`, the two dependent epics and `story:observatory-live-event-stream`. I traced about 25 acceptance reads to a producer in the set: 12 on a direct edge and 13 on a transitive `depends_on` path, none unrecorded.

What I could not establish:
- Out of my lane, for parallel safety: `inspect-device` and `checkpoint-cpu` both add `serde` and `serde_json` to root `Cargo.toml` with no edge between them.
- Out of my lane, for scope: `inspect-device` lists `crates/modelspec-docs/src/status.rs` in Scope, and no sentence of its body changes it.
- I did not read the other critics' round-2 files, though `m1-scope-round-2.md` and `m1-acceptance-round-2.md` are visible in the review-result directory.

```findings
- file: .engineering/planning/story/m1-vulkan-reduce-matmul.md
  line: 163
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the predicate 'device reports all three float-control values' is decided in three places (inspect-device negotiate reports a class, this story's emitter and its printed class re-read the three booleans, train RunStats names a class) and no body says the emitter and the printed class use the one negotiate result, so the module modes and the reported class can disagree"
- file: .engineering/planning/story/m1-compile-artifact.md
  line: 60
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "compile --backend vulkan takes a DeviceProfile (from --device-profile or the probe) and emits vulkan1.2 modules keyed to it but never applies the E-TARGET-004 version gate that inspect-device puts in negotiate, so a profile below 1.2 compiles to modules its device cannot run; the body must say compile runs negotiate and acceptance 5 needs a 1.1.0 profile case"
- file: .engineering/planning/story/m1-kernel-plan.md
  line: 94
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the step plan now holds an Allocate step for the partials buffer while vulkan-train records the step plan once per step, and no body says whether Allocate runs once per run or per recording, so the replay contract of the plan is split between kernel-plan, reduce-matmul and train; the body must state that step-plan Allocate steps are hoisted to run setup and that the verifier checks it"
```
