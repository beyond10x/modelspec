---
format: aep.planning-md/3
id: executable-system-specification:modelspec-contracts
kind: executable-system-specification
status: validated
title: 'ModelSpec data contracts: artifact, checkpoint, device, training run'
revision: 4
transitions:
- {from: "draft", to: "validated", at: "2026-10-21T08:30:06Z", actor: "human:timo", revision: 2}
---
# ModelSpec data contracts

## What it specifies

The data contracts between the compiler, the runtime and the files they exchange, drafted from
DESIGN.md:

| Domain | File | Source |
| --- | --- | --- |
| `modelspec.artifact` | `spec/domains/artifact.yaml` | DESIGN.md § 2, § 2.3, § 3, § 3.1, § 3.3, § 5.3, § 6.1, § 6.2, § 7, § 7.1 |
| `modelspec.checkpoint` | `spec/domains/checkpoint.yaml` | DESIGN.md § 2.2, § 2.3, § 6.1, § 6.2, § 9 |
| `modelspec.device` | `spec/domains/device.yaml` | DESIGN.md § 0, § 3.1, § 5.1, § 5.2, § 8, § 8.1 |
| `modelspec.training` | `spec/domains/training.yaml` | DESIGN.md § 1, § 2.3, § 6, § 6.1, § 6.2, § 9 |

The `.mspec` language itself (grammar, types, shapes, gradient rules) is not an ESS domain; it stays
in DESIGN.md § 2–4 and is held by the language and gradient tests of DESIGN.md § 9.

## Validation

From inside `spec/`: `ess specify validate --path .` → `modelspec v1 — 5 file(s), valid` (ess 0.66.0,
pinned by `spec/ess-inputs.yaml`). No `UNMAPPED:` marker remains.

## Decisions taken (2026-10-21)

The six questions the first draft left open, settled by the operator's delegate:

1. `CompiledArtifact` manifest = `compiler_version`, `schema_hash`, `parameters`
   (`ParameterSpec`: path, dtype, shape); plan = ordered `List<PlanStep>`, each an execution-DAG op
   (`Allocate`, `Copy`, `Dispatch`, `Barrier`, `Complete`) and, for a dispatch, its kernel module
   (DESIGN.md § 3.1, § 6.1, § 7.1).
2. RNG state = `RngState` (algorithm, seed, counter) with the algorithm fixed as Philox4x32-10;
   DESIGN.md § 2.3 now names it (revision 0.2). Optimizer state = `OptimizerSlot` list (path, slot,
   offset, length), empty for plain SGD.
3. A checkpoint does not reference the training run that wrote it: DESIGN.md § 6.2 makes it loadable
   on its own, and it carries `schema_hash`, `step`, optimizer and RNG state itself.
4. Device profile adds `vulkan_api_version`, `spirv_version`, `max_storage_buffer_range`,
   `min_storage_buffer_offset_alignment` and `cooperative_matrix_shapes`
   (`VkCooperativeMatrixPropertiesKHR` m, n, k and component types).
5. A training run references its `DeviceProfile` through an optional `device_profile_id`, absent
   for a CPU run.
6. A training run records `optimizer` (`OptimizerConfig`: kind `Sgd`, `learning_rate` as
   `Binary64`) and the `rng` state it started from.

## Decisions from design review 1 (2026-10-21)

Taken on the findings of `.engineering/drafts/ess-design-review-1.md`; DESIGN.md revision 0.3 carries
D3, D4, D6 and D8–D10.

- D1. `TrainingRun.seed` is removed; `rng.seed` is the single seed (DESIGN.md § 6.1).
- D2. `Checkpoint.optimizer` (hyperparameters) stays: resuming from a checkpoint needs them, and
  § 6.2 makes a checkpoint loadable on its own (DESIGN.md § 6.1, § 9 gate 7).
- D3. `deterministic` is a compile setting on `CompiledArtifact`: it changes reduction order, so it
  is part of the artifact hash; a training run reads it through its artifact (DESIGN.md § 3.3, § 6.1).
- D4. `ParameterSpec.canonical_parameter_id` is the first 8 bytes of blake3 of the canonical path,
  little-endian, as an unsigned 64-bit integer (DESIGN.md § 2.3).
- D5. `TrainingRun.checkpoint_every` is an optional positive integer; absent means no periodic
  checkpoints (DESIGN.md § 6).
- D6. A Cpu-backend `CompiledArtifact` has no kernels and no Dispatch plan steps (DESIGN.md § 7.1);
  the plan half is an invariant, the kernels half a summary.
- D7. `TrainingRun.compiler_version` is removed; the artifact carries it and the run reads it
  through its artifact, as D1 and D3 (DESIGN.md § 6.1).
- D8. `Checkpoint.header_checksum` is blake3 over the header and manifest bytes, the checksum field
  itself counted as zero bytes (DESIGN.md § 6.2).
- D9. Tensor offsets in a checkpoint are absolute file offsets (DESIGN.md § 6.2).
- D10. `TrainingRun.steps` counts the steps of this invocation, so a run ends at
  `start_step + steps`; `current_step <= start_step + steps` is a summary, not an invariant (L11)
  (DESIGN.md § 6).

## ESS limits

Rules DESIGN.md states that ESS 0.66.0 refuses as invariants; each stays in a `summary`.

- L1. `offset + length <= file_size`: arithmetic between two facts is refused (`type_mismatch`).
- L2. ParameterRecord bounds against its checkpoint: an identity cannot select another entity's field (`unobservable_fact`).
- L3. `offset % alignment == 0` (checkpoint tensors, BufferView): `%` is refused (`unparsable_predicate`).
- L4. `length == product(shape) * dtype.size_bytes`: lists have no product (`unobservable_fact`).
- L5. Unique `(path, slot)` among OptimizerSlots: `distinct` takes one key (`unparsable_predicate`).
- L6. Unique ParameterRecord path per checkpoint: an invariant cannot read owned relation rows (`unobservable_fact`).
- L7. Checkpoint loads only where schema_hash and parameter set match, and `start_step == checkpoint.step`: relation reads are refused (`unobservable_fact`).
- L8. A Dispatch's `kernel_module_id` names a kernel of the same artifact: `kernels` is not observable (`unobservable_fact`).
- L9. `device_profile_id` present iff `artifact.backend == Vulkan`: cross-entity read refused (`unobservable_fact`).
- L10. A Cpu artifact has no kernels: `kernels.count` is not observable (`unobservable_fact`).
- L11. `current_step <= start_step + steps`: arithmetic between two facts is refused (`type_mismatch`, ESS-ENTITY-002).

No commands, events or views yet, so `ess verify conform synthesize` yields no scenarios.
