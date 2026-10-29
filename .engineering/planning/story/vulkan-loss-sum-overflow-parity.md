---
format: aep.planning-md/3
id: story:vulkan-loss-sum-overflow-parity
kind: story
status: active
title: The device loss sum overflows no earlier than the CPU loss
summary: The Vulkan loss sum turns non-finite at the step the CPU's f64 mean does, restoring the DESIGN.md § 8.1 promise broken since 0.21.0.
relations:
- decomposes: epic:m1-vulkan-execution
- serves: vision:modelspec
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T08:47:22Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T08:47:23Z", actor: "human:timo", revision: 3}
---
# The device loss sum overflows no earlier than the CPU loss

## Context

Since 0.21.0 (a123c1a5, decision D84) the CPU adds up a batch's `mse` loss in f64 and the device
in f32 (DESIGN.md § 5.4). On `examples/` with 8 rows at lr 0.6 the device's f32 sum overflows at
step 372 while the CPU's f64 mean stays finite until step 382. DESIGN.md § 8.1 promises that a
device run names the step the CPU names when a loss turns non-finite; this breaks it. Found by
the record-branch test fix 4e26ab91 (2026-10-29), which moved the affected cases to a 1-row
example that overflows at step 372 on both. Accepted for the step-time wave as decision D143.

## Outcome

A device run turns non-finite at the step the CPU run does, on the 8-row example at lr 0.6 and on
every case of `train_vulkan_non_finite` and its adversary, with the 8-row cases restored.

## Options (decide in DESIGN.md with the change)

- device loss sum in f64 (shaderFloat64 on the RTX 3090), or
- scale each row loss by 1/n before the f32 sum, so the sum overflows only where the mean does.

## Acceptance

- `train_vulkan_non_finite`: an 8-row lr 0.6 case names step 382 on the CPU and on Vulkan, red first.
- Training bits otherwise unchanged; the run tolerance of DESIGN.md § 9 gate 5 still holds.
