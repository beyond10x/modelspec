---
format: aep.planning-md/3
id: decision-blocker:cooperative-matrix-f32-operands
kind: decision-blocker
status: cleared
title: The RTX 3090 reports no f32-operand cooperative-matrix shape; ADR-008 forbids reduced-precision operands
relations:
- blocks: story:fast-math-cooperative-matrix
withholds: test_result
revision: 5
transitions:
- {from: "open", to: "cleared", at: "2026-10-24T18:58:25Z", actor: "human:timo", revision: 5}
---
# The RTX 3090 reports no f32-operand cooperative-matrix shape; ADR-008 forbids reduced-precision operands

## The fact

`modelspec inspect-device --backend vulkan --format json` on 2026-10-24 (driver 610.57.04,
`10de:2204`) lists eleven `VK_KHR_cooperative_matrix` shapes. Their operand types are
`FLOAT16`, `BFLOAT16`, `SINT8` and `UINT8`; `FLOAT32` appears only as the accumulator and result
type (`c_type`, `result_type`). ADR-008 (DESIGN.md § 12, accepted 2026-10-24) permits a
cooperative-matrix matmul and says "no TF32 or other reduced-precision accumulation; every sum
accumulates in `f32`", and the benchmark rule compares against PyTorch eager CUDA with TF32 off.
An `f16` or `bf16` operand rounds every input to 11 or 8 significant bits before the product, a
larger change than TF32's 11-bit multiply, so it is outside the ADR as written.

## The decision, which is the operator's

| option | what it does | what it costs |
|---|---|---|
| A | amend ADR-008 to admit `f16` or `bf16` operands with `f32` accumulation under `fast_math`, compared within a tolerance the amendment states | an ADR change; the `fast_math` figure is then not comparable with PyTorch's FP32 figure and must be labelled as mixed precision |
| B | keep ADR-008 as accepted; story:fast-math-cooperative-matrix waits for a device that reports an `f32`-operand shape | no cooperative-matrix path on the RTX 3090; the FP32 path (contraction, transposed operands, larger tiles, fusion) carries the batch-8192 goal |

Until cleared, story:fast-math-cooperative-matrix has no `test_result` to record on this device.
Wave coordinator, 2026-10-24, decision D12 on `.engineering/waves/perf-1.md`.

Decider: operator

## Cleared (2026-10-24)

Option A taken by the coordinator under Timo's grant of 2026-10-24 ("You have all authority now to make
these decissions yourself"): ADR-008 amended (DESIGN.md § 12, revision 0.35): under `fast_math` a
cooperative-matrix matmul may take `f16` or `bf16` operands with `f32` accumulation, compared within
tolerance against an oracle computing with the same rounded operands, and benchmarked only against a
PyTorch baseline of the same operand precision, never as the headline. Decision D17 on
`.engineering/waves/perf-1.md`.
