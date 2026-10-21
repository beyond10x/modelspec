---
format: aep.planning-md/3
id: executable-system-specification:modelspec-contracts
kind: executable-system-specification
status: validated
title: 'ModelSpec data contracts: artifact, checkpoint, device, training run'
revision: 2
transitions:
- {from: "draft", to: "validated", at: "2026-10-21T08:30:06Z", actor: "human:timo", revision: 2}
---
# ModelSpec data contracts

## What it specifies

The data contracts between the compiler, the runtime and the files they exchange, drafted from
DESIGN.md:

| Domain | File | Source |
| --- | --- | --- |
| `modelspec.artifact` | `spec/domains/artifact.yaml` | DESIGN.md § 3, § 5.3, § 7.1 |
| `modelspec.checkpoint` | `spec/domains/checkpoint.yaml` | DESIGN.md § 2.3, § 6.2 |
| `modelspec.device` | `spec/domains/device.yaml` | DESIGN.md § 5.1, § 8 |
| `modelspec.training` | `spec/domains/training.yaml` | DESIGN.md § 6, § 6.1 |

The `.mspec` language itself (grammar, types, shapes, gradient rules) is not an ESS domain; it stays
in DESIGN.md § 2–4 and is held by the language and gradient tests of DESIGN.md § 9.

## Validation

`ess specify validate --path spec` → `modelspec v1 — 5 file(s), valid` (ess 0.66.0, pinned by
`spec/ess-inputs.yaml`).

## Open (`UNMAPPED:` markers)

1. `artifact.yaml`: the fields of the artifact `manifest` and `plan`.
2. `checkpoint.yaml`: the shape of RNG state and optimizer state descriptors.
3. `checkpoint.yaml`: whether a checkpoint references the training run that wrote it.
4. `device.yaml`: feature levels, cooperative-matrix shapes, storage-buffer limits.
5. `training.yaml`: the device link of a run (CPU runs have none).
6. `training.yaml`: the shape of optimizer hyperparameters and RNG state.

No commands, events or views yet, so `ess verify conform synthesize` yields no scenarios.
