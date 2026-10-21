---
format: aep.planning-md/3
id: epic:m1-vulkan-execution
kind: epic
status: draft
title: 'M1: native Vulkan execution on the RTX 3090'
relations:
- depends_on: epic:m0-semantic-core
- informed_by: executable-system-specification:modelspec-contracts
revision: 1
---
# M1: native Vulkan execution

## Outcome

The first end-to-end f32 training on an RTX 3090 through a Rust-owned Vulkan runtime and
generated SPIR-V, without CUDA API or toolkit (DESIGN.md § 1, row M1; § 11, iteration 2).

## In scope

- Headless `ash` runtime: device discovery, feature negotiation, allocator, dispatch, sync
  (DESIGN.md § 6).
- `rspirv` emission of elementwise, reduction, transpose, naive matmul and SGD kernels
  (DESIGN.md § 5.1, § 5.2 "M1 reference GPU kernels").
- `.msckpt` save/load (DESIGN.md § 6.2); `inspect-device`, `compile`, `train`, `infer` CLI verbs.

## Acceptance

The nine checkboxes of DESIGN.md § 9 "M1 acceptance criteria", unchanged.

## Specification

`CompiledArtifact`, `KernelModule`, `Checkpoint`, `ParameterRecord` and `DeviceProfile` are typed in
`spec/domains/` (artifact `executable-system-specification:modelspec-contracts`). Their `UNMAPPED:`
markers are settled before stories are written around them.
