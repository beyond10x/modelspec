---
format: aep.planning-md/3
id: approval-record:m1-tolerance-numerics-class
kind: approval-record
status: draft
title: M1 on the tolerance numerics class, per-op tolerances
relations:
- serves: vision:modelspec
revision: 1
---
# M1 runs in the tolerance numerics class on the RTX 3090

- Observed 2026-10-22: the RTX 3090, driver 610.57.04, reports
  `shaderDenormPreserveFloat32 = false` (`modelspec inspect-device` in story `m1-inspect-device`,
  and `vulkaninfo` read by b10x-chef-orchestrator at 06:24).
- Under DESIGN.md § 5.4 that puts the device in the tolerance numerics class: no GPU result on it
  is bit-for-bit with the CPU oracle, so ADR-007's bit-for-bit clause has no case on the target
  device.
- Decision, accepted by b10x-chef-orchestrator on 2026-10-22: M1 continues on the tolerance class;
  DESIGN.md § 5.4 and ADR-007 stay unchanged.
- Condition: every GPU test compares against the CPU oracle with a tolerance named per operator in
  the test, not one global epsilon. This holds for `m1-vulkan-elementwise`,
  `m1-vulkan-reduce-matmul` and `m1-vulkan-train`, and is recorded before any of them merges.
- Source: the cross-session message from b10x-chef-orchestrator.
