---
format: aep.planning-md/3
id: review-result:ess-design-review-1
kind: review-result
status: active
title: ESS hardening, design review of the data contracts, round 1
relations:
- reviews: executable-system-specification:modelspec-contracts
revision: 1
---
**Verdict:** the spec validates (`modelspec v1 — 5 file(s), valid`, ess 0.66.0). Against DESIGN.md rev 0.2 it has 30 findings: 22 missing, 0 contradicts, 2 stale mapping, 2 spec-only, 4 unclear. The checkpoint file layout (§ 6.2) and the kernel cache key (§ 5.3) have the largest gaps. I edited no file.

Technique: ess:hardening technique 8 (design review). Control run: on a copy of `spec/` with `Checkpoint.step` deleted, the same brief returned it as `missing`, citing DESIGN.md:331 "RNG and step state" and checkpoint.yaml:59-61 — the technique fails on a planted defect.

| # | classification | design (file:line, quoted) | spec (file:line, quoted) | note |
|---|---|---|---|---|
| 1 | missing | DESIGN.md:330 "header: magic `MSCK`, format version, endian marker, manifest size" | spec/domains/checkpoint.yaml:46 "name: format_version" | Of the header, only the version is declared. Magic, endian marker and manifest size have no field. |
| 2 | missing | DESIGN.md:330 "checksum/hash;" | spec/domains/checkpoint.yaml:53-54 "payload_hash … Optional<…ContentHash>" | The header checksum is required. `payload_hash` is the optional payload hash from :333, a different thing. |
| 3 | missing | DESIGN.md:335 "Validate offset and size ranges against file size before mapping/allocating." | spec/domains/checkpoint.yaml:94-96 "offset >= 0", "length > 0" | No payload or file size field exists, so `offset + length <= size` cannot be stated. |
| 4 | missing | DESIGN.md:332 "contiguous little-endian tensor payload with explicit alignment" | spec/domains/checkpoint.yaml:90-91 "name: offset" | No alignment field and no `offset % alignment == 0` invariant. Contiguity and non-overlap are also undeclared. |
| 5 | missing | DESIGN.md:335 "never silently reinterpret tensor bytes" | spec/domains/checkpoint.yaml:92-93 "name: length / type: Integer" | `length` is not tied to shape × dtype size, and its unit (bytes or elements) is not stated. |
| 6 | missing | DESIGN.md:335 "Validate offset and size ranges" | spec/domains/checkpoint.yaml:24-36 "OptimizerSlot … offset … length" | Optimizer slots have no range invariants and no dtype or shape, unlike ParameterRecord. |
| 7 | missing | DESIGN.md:117 "Each parameter has a **stable canonical path**"; :118 "accidental aliasing is disallowed" | spec/domains/checkpoint.yaml:84-85 "name: path" | Nothing makes `path` unique within a checkpoint. The same gap exists in artifact.yaml:77-79 `parameters`. |
| 8 | missing | DESIGN.md:335 "Parameter-path migration between incompatible model definitions requires an explicit adapter" | spec/domains/checkpoint.yaml:48-50 "schema_hash … Compiler/model schema hash" | No rule says a checkpoint loads only where `schema_hash` and the parameter set match. artifact.yaml:76 states the rule only in a summary. |
| 9 | missing | DESIGN.md:102 "each dimension is a positive compile-time integer" | spec/domains/checkpoint.yaml:88-89 "shape / List<Integer>" | No positivity invariant on dimensions. artifact.yaml:33-34 has the same gap. |
| 10 | missing | DESIGN.md:220 "`fast_math = true` … is recorded in the artifact hash"; :387 "`blake3` for stable artifact … hashing" | spec/domains/artifact.yaml:68-70 "fast_math … Recorded in the artifact hash" | No artifact hash field exists; the identity at :56-58 is a Uuid. The summary points at a declaration that is not there. |
| 11 | missing | DESIGN.md:285 "Kernel configuration cache keys include IR hash, shapes, dtype, optimization settings, GPU/vendor/driver identifiers, and extension/capability set." | spec/domains/artifact.yaml:65-67 "ir_hash … Part of every kernel cache key" | No cache-key type or entity exists, and the artifact has no link to the device or capability set it was compiled for. |
| 12 | missing | DESIGN.md:158 "dispatch and copy commands, resource dependencies/barriers, scratch allocations, and CPU-visible completion points" | spec/domains/artifact.yaml:42-50 "PlanStep … op … kernel_module_id" | Steps name no resources, dependencies or allocation sizes, and no dispatch dimensions. The buffer-view fields of DESIGN.md:399 are also absent. |
| 13 | missing | DESIGN.md:397 "Untrusted serialized artifacts are validated before dispatch." | spec/domains/artifact.yaml:49-50 "kernel_module_id / Optional<…>" | No invariant that a Dispatch step has `kernel_module_id` and other steps do not, or that the id belongs to the same artifact's `kernels`. |
| 14 | missing | DESIGN.md:256 "FP16/storage features"; :434 "no supported f16 storage/compute path" | spec/domains/device.yaml:59-60 "f16_storage / Boolean" | f16 compute (arithmetic) support is not recorded. |
| 15 | missing | DESIGN.md:256 "subgroup properties" | spec/domains/device.yaml:55-56 "subgroup_size" | Only the size is recorded. Supported stages and operations are absent. |
| 16 | missing | DESIGN.md:256 "workgroup and storage-buffer limits"; :270 "fit shared memory" | spec/domains/device.yaml:57-58 "max_workgroup_invocations" | Per-dimension workgroup size, workgroup count and shared-memory size are absent. |
| 17 | missing | DESIGN.md:256 "memory heaps/budgets" | spec/domains/device.yaml:10-16 "MemoryHeap … size_bytes … device_local" | Budgets are not recorded. |
| 18 | missing | DESIGN.md:277 "enumerate *actual supported* matrix types and M/N/K shapes" | spec/domains/device.yaml:80-82 "Empty when `cooperative_matrix` is false" | The rule is stated only in a summary; there is no invariant. |
| 19 | missing | DESIGN.md:324 "Record source/compiler version" | spec/domains/training.yaml:54-55 "compiler_version" | No source version or source hash is recorded, in the run or in the artifact. |
| 20 | missing | DESIGN.md:324 "tensor shapes" | spec/domains/artifact.yaml:77-79 "parameters … shapes" | Only parameter shapes are recorded. Input and target shapes (DESIGN.md:74-75) are not. |
| 21 | missing | DESIGN.md:324 "step number"; :454 "uninterrupted and resumed **CPU** run equivalence" | spec/domains/training.yaml:58-60 "steps … TrainConfig.steps" | The run records only the step total. There is no start or current step, so a resumed run cannot be reproduced. |
| 22 | missing | DESIGN.md:324 "backend/device/driver" | spec/domains/training.yaml:69-71 "Absent for a CPU run" | No invariant ties `device_profile_id` being present to `artifact.backend == Vulkan`. |
| 23 | stale mapping | DESIGN.md:331 "parameter names, shapes, dtypes" (§ 6.2) | spec/domains/artifact.yaml:79 "parameter paths, dtypes and shapes (DESIGN.md § 6.1)" | § 6.1 (:324) does not mention dtypes. The list is in § 6.2, and that list describes the checkpoint manifest. |
| 24 | stale mapping | DESIGN.md:220 (§ 3.3), :330-331 (§ 6.2), :277 (§ 5.2), :36 (§ 1) | .engineering/drafts/ess.md:10-13 source table | The table leaves out sections the domains cite: artifact cites § 3.3 and § 6.2 (artifact.yaml:70, :76), device cites § 5.2 (device.yaml:63), training cites § 1 and § 6.2 (training.yaml:13, :35). |
| 25 | spec-only | DESIGN.md:331 lists "optimizer state descriptors", not hyperparameters | spec/domains/checkpoint.yaml:56-57 "optimizer / OptimizerConfig" | § 6.2 does not ask for optimizer hyperparameters in the checkpoint. The field rests only on ess.md decision 3. |
| 26 | spec-only | DESIGN.md:324 "RNG algorithm/seed/state" (one item) | spec/domains/training.yaml:56-57 "seed" next to :66-68 "rng" | The seed is recorded twice, with no invariant `seed == rng.seed`. |
| 27 | unclear | DESIGN.md:220 "reassociate floating-point reductions in deterministic mode"; :324 "`deterministic` mode disables nondeterministic algorithms" | spec/domains/training.yaml:61-63 "deterministic" | It is not decided whether deterministic is a compile flag (artifact, which would put it in the hash) or a run setting. |
| 28 | unclear | DESIGN.md:119 "`global_seed + canonical_parameter_id`"; :387 "parameter-identity hashing" | spec/domains/artifact.yaml:24-34 "ParameterSpec … path" | It is not stated whether `canonical_parameter_id` is the path or a derived hash. The spec has no id field. |
| 29 | unclear | DESIGN.md:307 "`checkpoint_every: Option<u64>`"; :320 "This interface is illustrative" | spec/domains/training.yaml:58-60 "TrainConfig.steps" | The spec cites `TrainConfig` for `steps` but omits `checkpoint_every`. The block is marked illustrative. |
| 30 | unclear | DESIGN.md:396 "The compiler emits a `CompiledArtifact { manifest, kernels, plan }`"; :19 "CPU reference execution … consume the same verified tensor IR" | spec/domains/artifact.yaml:20-22 "variants: [Cpu, Vulkan]" | The design does not say what a CPU artifact contains (kernels? plan?), so `Backend: Cpu` cannot be checked against it. |

**Counts:** missing 22, contradicts 0, stale mapping 2, spec-only 2, unclear 4.

**Not reached:**
- § 2–4 (language) were skipped as out of scope.
- § 8.1 diagnostics taxonomy (:440) and § 9 performance metrics (:457) are not data contracts of the four domains, so I did not check them.
- § 10–12 I read as context only.

A `device` relation with cardinality `one` on an optional field (training.yaml:78-82) is not a finding: ESS here offers only `one` or `many`.

```findings
- {file: spec/domains/checkpoint.yaml, line: 46, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Checkpoint declares no header magic MSCK, endian marker or manifest size (DESIGN.md:330)."}
- {file: spec/domains/checkpoint.yaml, line: 53, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "The mandatory header checksum/hash of DESIGN.md:330 is missing; payload_hash is the optional payload hash of :333."}
- {file: spec/domains/checkpoint.yaml, line: 94, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "No payload or file size field exists, so offset+length cannot be bounded as DESIGN.md:335 requires."}
- {file: spec/domains/checkpoint.yaml, line: 90, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "No alignment field or offset-alignment invariant for the explicitly aligned payload of DESIGN.md:332."}
- {file: spec/domains/checkpoint.yaml, line: 92, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "ParameterRecord.length is not tied to shape and dtype size and its unit is unstated, so bytes could be reinterpreted against DESIGN.md:335."}
- {file: spec/domains/checkpoint.yaml, line: 24, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "OptimizerSlot carries offset and length with no range invariants and no dtype or shape (DESIGN.md:335)."}
- {file: spec/domains/checkpoint.yaml, line: 84, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Parameter path is not unique within a checkpoint or within artifact.parameters (DESIGN.md:117-118)."}
- {file: spec/domains/checkpoint.yaml, line: 48, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "No rule restricts loading a checkpoint to an artifact with equal schema_hash and parameter set (DESIGN.md:335, :470)."}
- {file: spec/domains/checkpoint.yaml, line: 88, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Shape dimensions in ParameterRecord and ParameterSpec lack the positivity invariant of DESIGN.md:102."}
- {file: spec/domains/artifact.yaml, line: 68, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "No artifact hash field exists although fast_math must be recorded in it (DESIGN.md:220, :387)."}
- {file: spec/domains/artifact.yaml, line: 65, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "The kernel configuration cache key of DESIGN.md:285 (IR hash, shapes, dtype, settings, GPU/vendor/driver ids, capability set) is not declared."}
- {file: spec/domains/artifact.yaml, line: 42, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "PlanStep carries no resource dependencies, scratch allocation or dispatch dimensions required by the execution DAG of DESIGN.md:158."}
- {file: spec/domains/artifact.yaml, line: 49, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "No invariant that kernel_module_id is present exactly on Dispatch steps and names a kernel of the same artifact (DESIGN.md:397)."}
- {file: spec/domains/device.yaml, line: 59, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "DeviceProfile records f16 storage but not f16 compute support (DESIGN.md:256, :434)."}
- {file: spec/domains/device.yaml, line: 55, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Only subgroup_size is recorded of the subgroup properties DESIGN.md:256 requires."}
- {file: spec/domains/device.yaml, line: 57, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Workgroup limits stop at max_workgroup_invocations; per-dimension size, count and shared-memory size are missing (DESIGN.md:256, :270)."}
- {file: spec/domains/device.yaml, line: 10, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Memory budgets of DESIGN.md:256 are not recorded."}
- {file: spec/domains/device.yaml, line: 80, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "The rule that cooperative_matrix_shapes is empty when cooperative_matrix is false lives only in a summary, not an invariant (DESIGN.md:277)."}
- {file: spec/domains/training.yaml, line: 54, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "No source version or source hash is recorded although DESIGN.md:324 requires source/compiler version."}
- {file: spec/domains/artifact.yaml, line: 77, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Only parameter shapes are recorded; input and target tensor shapes required by DESIGN.md:324 are not."}
- {file: spec/domains/training.yaml, line: 58, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "TrainingRun has no start or current step number (DESIGN.md:324, :454), so a resumed run is not reproducible from the record."}
- {file: spec/domains/training.yaml, line: 69, category: spec-gap, severity: warning, verdict: NEEDS-CHANGE, origin: pre-existing, message: "No invariant ties device_profile_id presence to artifact.backend == Vulkan (DESIGN.md:324)."}
- {file: spec/domains/artifact.yaml, line: 79, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Stale mapping: the parameters summary cites DESIGN.md § 6.1 for dtypes, which appear only in § 6.2 (:331)."}
- {file: .engineering/drafts/ess.md, line: 10, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "Stale mapping: the source table omits sections the domains cite (§ 3.3 and § 6.2 for artifact, § 5.2 for device, § 1 and § 6.2 for training)."}
- {file: spec/domains/checkpoint.yaml, line: 56, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "spec-only: Checkpoint.optimizer hyperparameters have no basis in DESIGN.md:331, which lists only optimizer state descriptors."}
- {file: spec/domains/training.yaml, line: 56, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "spec-only: TrainingRun.seed duplicates rng.seed with no invariant and no design statement for two seeds (DESIGN.md:324)."}
- {file: spec/domains/training.yaml, line: 61, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "unclear: DESIGN.md:220 and :324 do not say whether deterministic is a compile flag (artifact hash) or a run setting."}
- {file: spec/domains/artifact.yaml, line: 24, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "unclear: DESIGN.md:119 canonical_parameter_id may be the path or a blake3 identity (:387), and the spec declares no id."}
- {file: spec/domains/training.yaml, line: 58, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "unclear: TrainConfig.checkpoint_every (DESIGN.md:307) is omitted while steps is cited, and the block is marked illustrative (:320)."}
- {file: spec/domains/artifact.yaml, line: 20, category: spec-gap, severity: note, verdict: NEEDS-CHANGE, origin: pre-existing, message: "unclear: DESIGN.md:396 and :19 do not say what a Cpu-backend CompiledArtifact contains, so Backend Cpu cannot be checked."}
```
