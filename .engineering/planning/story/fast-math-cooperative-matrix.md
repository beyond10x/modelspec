---
format: aep.planning-md/3
id: story:fast-math-cooperative-matrix
kind: story
status: draft
title: A cooperative-matrix matmul under fast_math
relations:
- serves: vision:modelspec
- depends_on: story:fast-math-mode
revision: 1
---
# A cooperative-matrix matmul under `fast_math`

## Why

Split from story:fast-math-mode on 2026-10-24 (decision D12 on `.engineering/waves/perf-1.md`).
ADR-008 permits a matmul to run as a cooperative-matrix kernel where the device reports
`VK_KHR_cooperative_matrix` (DESIGN.md § 12), with every sum accumulated in `f32` and no
reduced-precision operand. The RTX 3090 reports eleven shapes (`inspect-device --backend vulkan
--format json`, 2026-10-24): `a_type`/`b_type` are `FLOAT16`, `BFLOAT16`, `SINT8` or `UINT8`,
and `FLOAT32` appears only as `c_type`/`result_type`. No shape takes `f32` operands, so this
kernel cannot run on the project's device under ADR-008 as accepted.

Outcome: compiled with `--fast-math` for a device profile that reports an `f32`-operand
cooperative-matrix shape, a matmul whose dimensions are multiples of that shape runs as a
cooperative-matrix kernel; for a profile without such a shape the plan is the contraction-only
plan of story:fast-math-mode.

## Acceptance

- **shape-selected-from-profile**: the kernel plan names the cooperative-matrix shape it uses, taken
  from the `DeviceProfile`'s `cooperative_matrix_shapes` with `f32` `a_type`, `b_type`, `c_type` and
  `result_type`; a profile with no such shape yields no cooperative-matrix kernel, and a profile
  with `cooperative_matrix` false yields none.
- **cooperative-matrix-emitted**: compiled with `--fast-math` for a synthetic `--device-profile`
  that reports an `f32` shape, the matmul module declares the `CooperativeMatrixKHR` capability and
  the `SPV_KHR_cooperative_matrix` extension; without `--fast-math` or without the shape it does
  not.
- **accumulation-f32**: the emitted `OpCooperativeMatrixMulAddKHR` uses an `f32` accumulator
  type; a test reads it from the module.
- **device-run-or-refused**: on a device whose shapes do not include the kernel's, the runner refuses
  the artifact before dispatch with an `E-TARGET` code naming the shape; on a device that does, the
  run passes within ADR-008's tolerance against the CPU oracle.
- **throughput-reported** (reported, not gated): where a device with an `f32` shape is available,
  `mlp_sweep` at b8192 h4096 with the kernel beside the contraction-only figure.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-runtime/src/vulkan/dispatch.rs`
- `DESIGN.md`

## Out of scope

- `f16` or `bf16` operands with `f32` accumulation: ADR-008 forbids reduced-precision operands;
  admitting them is an ADR-008 amendment, which is the operator's.
