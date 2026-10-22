---
format: aep.planning-md/3
id: review-result:m1-acceptance-round-1
kind: review-result
status: active
title: Plan critic, acceptance lens, round 1, decomposition of epic:m1-vulkan-execution
relations:
- reviews: epic:m1-vulkan-execution
- reviews: story:m1-compile-artifact
- reviews: story:m1-vulkan-train
- reviews: story:m1-inspect-device
- reviews: story:m1-checkpoint-cpu
- reviews: story:m1-vulkan-profile
revision: 1
---
needs-revision

Findings:
story:m1-compile-artifact — item 5 depends on "a host without Vulkan", which the operator's RTX 3090 host is not, and it does not name `MODELSPEC_VULKAN_LIBRARY` the way story:m1-inspect-device item 2 does. The same check gives different answers on different hosts, and without `--device-profile` `compile --backend vulkan` would probe device 0, which is a Vulkan instance inside `task ci`. Quoted: "`compile --backend cpu` without `--device-profile` exits 0 on a host without Vulkan; `--backend vulkan` there exits 1 with E-TARGET-002". — .engineering/planning/story/m1-compile-artifact.md:109
story:m1-vulkan-train — item 6 has the same defect: "with it, on a host without a loader, it exits 1 with E-TARGET-002" names no way to produce a loader-less host. The acceptance cannot be checked in `task ci` or on the operator's machine. Story:m1-vulkan-checkpoint-infer item 5 does name the env var. — .engineering/planning/story/m1-vulkan-train.md:129
story:m1-inspect-device — items 1 and 6 read as ordinary CI checks, but they need `rtx3090.device.json`, which item 4 says only the operator run produces ("Its output is committed as the fixture"). Nothing says who creates the fixture or when, so item 1 and `task ci` cannot pass before the device run. The acceptance does not state that order, so "done" is unreadable until it is known. — .engineering/planning/story/m1-inspect-device.md:116 (fixture origin at :134)
story:m1-compile-artifact — item 4 expects code 003 for four altered artifacts. The reader checks `artifact_hash` before the invariants, and a plan edit changes the hash. Without a re-computed hash those cases return 002, and the acceptance does not say the test re-seals the hash. Quoted: "the reader refuses each of seven altered artifacts, with the code stated". — .engineering/planning/story/m1-compile-artifact.md:104
story:m1-checkpoint-cpu — item 4 requires "no allocation larger than the file" and names no way to observe an allocation, such as a counting allocator in the test. A program cannot get the same answer twice from it, so the check reduces to someone asserting it. — .engineering/planning/story/m1-checkpoint-cpu.md:124
story:m1-vulkan-profile — item 1 expects every module to show `calls` 1,800, or 1,801 when the forward plan also dispatches it. Module ids come from kernel content hashes, so two identical kernels in one step would share a module and show a different count. The acceptance does not say one module per dispatch, nor give a rule for a module used twice. — .engineering/planning/story/m1-vulkan-profile.md:84

Device-dependent items, read as asked (all artifacts):
- The operator-run items are checkable by the operator. Each names the command, the pass condition and a numeric bound or a count.
- I re-derived the arithmetic and it holds. Upload is 520,864 bytes and download 210,468 bytes. Submits are `ceil(1800/256) = 8`. There are 37 `step` lines (`LOSS_EVERY = 50`, `crates/modelspec-cli/src/train.rs:130`). The DeviceProfile has 21 fields besides the identity (`spec/domains/device.yaml`).
- The CI-side items consistently use the `MODELSPEC_VULKAN_LIBRARY` missing-path trick, except the two findings above.

What I read: 10 of 10 ids (the 10 drafts), each with `aep plan artifact show` in full. I also read `kinds` and `lifecycle story`, the parent epic, the decomposer report, DESIGN.md § 9 "M1 acceptance criteria", `story:bert-vulkan-kernels-rowwise` for the acceptance convention, `data.rs`, `gradients.rs` and `train.rs`.

What I could not establish:
- The checks on `calls` counts in vulkan-profile and on the 1,800-step GPU/CPU agreement (1e-4 L2, per-loss tolerance) need a device run. I did not judge whether they are attainable, which is outside my lane.
- vulkan-profile's no-timestamp-query fallback line is in "What it adds" but no acceptance item checks it. The RTX 3090 has timestamps and `task ci` has no Vulkan, so nothing can exercise it. This is an unease, not a finding.
- The "First:" ordering items (DESIGN edits before code) cannot be observed after the fact. I judged them a convention, not a defect.
- Out of my lane, did not set the verdict: the DESIGN § 3.3 vs § 5.2 reassociation conflict the decomposer flags (design). Overlapping scope on `crates/modelspec-runtime/src/spirv/emit.rs`, `vulkan/` and `tests/spirv_val.rs` across several stories (parallel-safety).

```findings
- file: .engineering/planning/story/m1-compile-artifact.md
  line: 109
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "item 5 depends on 'a host without Vulkan' and does not name MODELSPEC_VULKAN_LIBRARY, so it reads differently on the operator's GPU host than in CI, and compile --backend vulkan without --device-profile would create a Vulkan instance in task ci"
- file: .engineering/planning/story/m1-vulkan-train.md
  line: 129
  category: acceptance
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "item 6 expects E-TARGET-002 'on a host without a loader' and names no way to produce one (story:m1-vulkan-checkpoint-infer item 5 uses MODELSPEC_VULKAN_LIBRARY), so it is checkable neither in task ci nor on the operator's host"
- file: .engineering/planning/story/m1-inspect-device.md
  line: 116
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "items 1 and 6 pass only once rtx3090.device.json exists, which item 4 says only the operator device run produces; the acceptance does not state who creates the fixture or in what order"
- file: .engineering/planning/story/m1-compile-artifact.md
  line: 104
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "item 4 expects code 003 for four altered artifacts but the reader checks artifact_hash first, so unless the test re-computes the hash they return 002; the acceptance does not say the altered artifacts are re-sealed"
- file: .engineering/planning/story/m1-checkpoint-cpu.md
  line: 124
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "item 4 requires 'no allocation larger than the file' and names no observation mechanism (for example a counting allocator in the test), so it cannot be checked the same way twice"
- file: .engineering/planning/story/m1-vulkan-profile.md
  line: 84
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "item 1 expects every module to show calls 1,800 or 1,801 but module ids are content hashes and the acceptance does not say one module per dispatch or give the count for a module used by two dispatches"
```
