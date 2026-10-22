---
format: aep.planning-md/3
id: review-result:m1-design-round-1
kind: review-result
status: active
title: Plan critic, design lens, round 1, decomposition of epic:m1-vulkan-execution
relations:
- reviews: epic:m1-vulkan-execution
- reviews: story:m1-kernel-plan
- reviews: story:m1-vulkan-reduce-matmul
- reviews: story:m1-vulkan-elementwise
- reviews: story:m1-checkpoint-cpu
- reviews: story:m1-vulkan-train
- reviews: story:m1-inspect-device
- reviews: story:m1-compile-artifact
revision: 1
---
needs-revision

Decision 7 holds in substance but not as drafted. The M1 set is otherwise a sound shape: it has no cycle, it is not a queue, and no layer-cut story fails to stand on its own.

**Findings**

- story:m1-kernel-plan — Its acceptance fixes "exactly one Dispatch per computing op", one kernel per op and one buffer per value, but the reduce-matmul story needs a two-stage reduction. That is two dispatches, a barrier and a partials scratch buffer, and `KernelModule` has one `entry_point`, so it also needs two modules. The reduce-matmul scope has no `modelspec-core` file, so the barrier and scratch would exist only in runtime `dispatch.rs`, outside the verified plan. The fix is for the schedule to lower a reduce to stage 1, an `Allocate` scratch step, a barrier and stage 2, with the verifier and acceptance 2 saying so. — `.engineering/drafts/m1-kernel-plan.md:27`, `.engineering/drafts/m1-kernel-plan.md:82`, `.engineering/drafts/m1-vulkan-reduce-matmul.md:28`, `spec/domains/artifact.yaml:96`
- story:m1-vulkan-reduce-matmul — Decision 7 changes what `deterministic` means, but the scope names no spec file. `spec/domains/artifact.yaml:188` types it as "no floating-point reduction is reassociated". The spec-first rule requires that summary to be edited and `ess specify validate` run in the same change. — `.engineering/drafts/m1-vulkan-reduce-matmul.md:40`, `spec/domains/artifact.yaml:188`
- story:m1-vulkan-reduce-matmul — The rule is worded "a GPU reduction uses a fixed tree … not the CPU order", and bit-equality is "recorded, not asserted". Both conflict with the pending 0.8 patch, which puts a one-invocation row-major loss sum and non-split-K matmul in the bit-for-bit class and parallel reductions in the tolerance class. The fix is to word the rule as "an order fixed by shape alone, workgroup size a compile-time constant not read from DeviceProfile". It should keep the CPU order as one admissible order. The story should also say whether M1 emits the RTE, denorm-preserve and signed-zero execution modes and asserts bit-equality, or leaves that class to M2. — `.engineering/drafts/m1-vulkan-reduce-matmul.md:19`, `.engineering/drafts/m1-vulkan-reduce-matmul.md:64`, `~/.cache/modelspec-wave-cpu-perf-1/design-10x/design-10x.patch:50`, `~/.cache/modelspec-wave-cpu-perf-1/design-10x/design-10x.patch:51`
- story:m1-vulkan-elementwise — Its acceptance 1 writes the gate-5 reference `r` for "a reduction or a matmul element" before the reduction rule exists. That rule is settled in the reduce-matmul story, so each of the two stories writes half of one numerics contract. The elementwise story should state `r` for elementwise kernels only, and the reduce-matmul story should extend gate 5 for reductions and matmul. — `.engineering/drafts/m1-vulkan-elementwise.md:21`, `.engineering/drafts/m1-vulkan-elementwise.md:62`, `.engineering/drafts/m1-vulkan-reduce-matmul.md:40`
- story:m1-checkpoint-cpu — This is the DESIGN.md ownership trade-off. Eight of ten stories edit DESIGN.md and each bumps the one revision line. Five edit § 8: inspect-device, compile-artifact, vulkan-train, checkpoint-cpu and profile. checkpoint-cpu is the most unordered editor, with no path to inspect-device, elementwise, reduce-matmul, compile-artifact, vulkan-train or profile. Option A is to add `depends_on` edges from it to the other DESIGN editors, which delays the CPU checkpoint behind the GPU spine. Option B is to drop the revision bump and the § 8 text from the stories and give both to one design-owner change. — `.engineering/drafts/m1-checkpoint-cpu.md:62`, `.engineering/drafts/m1-vulkan-train.md:51`, `.engineering/drafts/m1-inspect-device.md:56`
- story:m1-vulkan-train — Its § 6 rewrite lands on the same sentence the 0.8 patch extends with "From M2 the runtime does not block per step either (§ 6.3)". The patch's § 6.3 items 4 and 5 describe as M2 what this story and checkpoint-infer build in M1: chunked recording, waits only at the end or at a checkpoint, and read-back. The body should say its edit applies on top of 0.8. It should also say that § 6.3 keeps only the M2 increments of device gather, parameter ping-pong and fused kernels. — `.engineering/drafts/m1-vulkan-train.md:16`, `.engineering/drafts/m1-vulkan-train.md:51`, `~/.cache/modelspec-wave-cpu-perf-1/design-10x/design-10x.patch:62`, `~/.cache/modelspec-wave-cpu-perf-1/design-10x/design-10x.patch:86`
- story:m1-inspect-device — The patch gates the bit-for-bit class on `shaderRoundingModeRTEFloat32`, `shaderDenormPreserveFloat32` and `shaderSignedZeroInfNanPreserveFloat32`. Its notes say "The M1 `inspect-device` reports them", but `spec/domains/device.yaml` has no such field and the draft prints 21 fields. Adding them later changes a typed record and invalidates the committed `rtx3090.device.json` fixture, so the three fields and their acceptance belong in this story now. — `.engineering/drafts/m1-inspect-device.md:85`, `~/.cache/modelspec-wave-cpu-perf-1/design-10x/design-10x-notes.md:33`, `spec/domains/device.yaml:72`
- story:m1-kernel-plan — `ParameterSpec.view` is a single view with an in-place update. Patch § 6.3 item 3 needs two versions per parameter, because the W2 update runs beside a kernel that still reads the old W2. M2 would therefore break a type this story pins first. The body should type the parameter's views so that one view is the in-place case and two is ping-pong. — `.engineering/drafts/m1-kernel-plan.md:22`, `.engineering/drafts/m1-kernel-plan.md:32`, `~/.cache/modelspec-wave-cpu-perf-1/design-10x/design-10x.patch:85`
- story:m1-compile-artifact — It bundles two things: the entry that emits every kernel of an artifact, which vulkan-train needs, and the on-disk format with its reader and E-ARTIFACT codes. No M1 story loads the file, because vulkan-train compiles in memory. The reader and codes therefore sit on the critical path before vulkan-train for no reader of theirs. Moving the emit-all entry into the reduce-matmul story's `emit.rs` would let vulkan-train depend on it directly. — `.engineering/drafts/m1-compile-artifact.md:42`, `.engineering/drafts/m1-vulkan-train.md:20`, `.engineering/drafts/m1-compile-artifact.md:26`

**Decision 7.** The direction is right: a shape-fixed two-stage order is the only way to keep § 5.2's two-stage kernel under `deterministic`. It is a design change, though, not a reconciliation. § 3.3 line 224 forbids reassociating a reduction, and a tree reassociates against the CPU order of § 2.2. The ESS summary at `artifact.yaml:188` says the same thing. Worded as the family "order fixed by shape alone", it is consistent with the 0.8 patch. It also needs the two findings above that it currently lacks: the spec edit and the plan representation.

**DESIGN.md ownership.** Eight of ten stories edit it, and only mlp-example and checkpoint-infer do not. The edits sit in § 3.1, § 3.3, § 5.2, § 6, § 6.2, § 7.1, § 8 and § 9 gate 5. Several section owners is right, because AGENTS.md puts a design edit in the same change as the code. Two surfaces need a single owner: the revision line, and the § 8 verb table that five stories append to. The numerics contract (gate-5 tolerances plus the reduction rule) also needs one owner, because it is split between elementwise and reduce-matmul today.

**Conflicts with the 10x patch.** The patch is based on DESIGN revision 0.7, which is the revision in the plan tree.
- **Revision line:** the patch sets 0.8, and each of the eight stories bumps the same line, so the order of application must be fixed.
- **§ 6 Execution:** the patch and vulkan-train edit the same sentence.
- **§ 5.4:** its bit-for-bit class applies to kernels M1 builds, but M1 neither emits the execution modes nor asserts the result.
- **Float-control features:** DeviceProfile does not carry the three the patch gates on.
- **Parameter views:** the patch needs two versions per parameter, and kernel-plan's type has one view.
- **No conflict:** the patch leaves the gate-5 number open, and M1 elementwise supplies it. The 20-step warm-up in the profile story matches the patch's timing rule.

**What I read:** 10 draft stories via `aep plan artifact show`. I ran `aep plan artifact relations`, `graph` and `validate` (valid, 84 artifacts). I also read the decomposer report, the patch and its notes, DESIGN.md §§ 0–9, `spec/domains/artifact.yaml` and `device.yaml`, and `train/data.rs`. I walked 38 edges, including the epic edges to `epic:m0-semantic-core`, `epic:bert-from-scratch` and `epic:pytorch-10x-mnist`, and found no needs-first cycle. The longest chain is 6 waves of 10 stories. I traced 17 acceptance reads to a producer in the set. Ten are direct `depends_on` edges and seven are covered by a transitive path, so none is unrecorded. The seven are the device fixture, `GPU_ATOL`, the sibling-story reads by the profile guide and the `mnist_mlp` reads of profile and checkpoint-infer.

**What I could not establish**
- I did not read the `m1-critic-*` files in `.engineering/drafts/`, by design.
- Whether a 128-invocation reduction workgroup (kernel-plan's cap) matches the elementwise workgroup of 64 is unstated in the reduce-matmul story. I took it as part of the wording finding.
- Out of my lane, for the acceptance critic: inspect-device acceptance 1 reads a fixture that its own operator run (acceptance 4) creates, and compile-artifact acceptance 1 reads the same fixture in `task ci`.
- Out of my lane, for the parallel-safety critic: `train/runner.rs` and `cli/train.rs` are edited by checkpoint-cpu, vulkan-train and checkpoint-infer. `core/src/artifact.rs` is marked "(new)" by both kernel-plan and compile-artifact. `spirv/emit.rs` is edited by elementwise, reduce-matmul and compile-artifact.

```findings
- file: .engineering/drafts/m1-kernel-plan.md
  line: 82
  category: design
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "the acceptance fixes exactly one Dispatch per computing op and one kernel and buffer per op, but the two-stage reduction of story:m1-vulkan-reduce-matmul needs two dispatches, a barrier, a partials scratch buffer and two modules, so the schedule must lower a reduce to stage 1, an Allocate step, a barrier and stage 2 in the verified plan"
- file: .engineering/drafts/m1-vulkan-reduce-matmul.md
  line: 40
  category: design
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "the scope names no spec file although the fixed reduction tree changes the meaning of deterministic typed at spec/domains/artifact.yaml:188 as no floating-point reduction is reassociated, so that summary must be edited and validated in the same story"
- file: .engineering/drafts/m1-vulkan-reduce-matmul.md
  line: 19
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the rule is worded as a fixed tree that is not the CPU order with bit-equality recorded not asserted, which conflicts with the pending 0.8 patch section 5.4 (one-invocation row-major sum and non-split-K matmul are bit-for-bit, parallel reductions are tolerance), and it does not say the workgroup size is a compile-time constant independent of DeviceProfile or whether M1 emits the float-control execution modes"
- file: .engineering/drafts/m1-vulkan-elementwise.md
  line: 21
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "acceptance 1 writes the gate-5 reference r for reductions and matmul before the reduction rule exists, so this story and story:m1-vulkan-reduce-matmul each write half of one numerics contract; state r for elementwise kernels only here and extend gate 5 for reductions and matmul in the reduce-matmul story"
- file: .engineering/drafts/m1-checkpoint-cpu.md
  line: 62
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "eight of ten stories edit DESIGN.md and each bumps the one revision line while five edit section 8, and this story is unordered against all of them except kernel-plan; either record depends_on edges among the DESIGN editors or move the revision bump and section 8 text to one design-owner change"
- file: .engineering/drafts/m1-vulkan-train.md
  line: 51
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the section 6 rewrite lands on the sentence the pending 0.8 patch extends with From M2 the runtime does not block per step, and patch section 6.3 items 4 and 5 describe as M2 what this story builds in M1, so the body must say its edit applies on 0.8 and that section 6.3 keeps only the M2 increments"
- file: .engineering/drafts/m1-inspect-device.md
  line: 85
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the 21 printed fields omit shaderRoundingModeRTEFloat32, shaderDenormPreserveFloat32 and shaderSignedZeroInfNanPreserveFloat32 that the 0.8 patch gates bit-for-bit on and whose notes say M1 inspect-device reports, so device.yaml, the probe, the acceptance and the fixture must carry them now"
- file: .engineering/drafts/m1-kernel-plan.md
  line: 22
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "ParameterSpec.view is typed as a single view with in-place update while patch section 6.3 item 3 needs two versions per parameter, so M2 would break a type this story pins first; type the parameter's views so one is the in-place case and two is ping-pong"
- file: .engineering/drafts/m1-compile-artifact.md
  line: 42
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the story bundles the emit-every-kernel entry that story:m1-vulkan-train needs with an on-disk format, reader and E-ARTIFACT codes that no M1 story loads, which puts the reader on the critical path before vulkan-train; move the emit entry into the reduce-matmul emit.rs so vulkan-train depends on it directly"
```
