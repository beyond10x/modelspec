---
format: aep.planning-md/3
id: story:vulkan-scalar-operand-in-reduce-and-loss
kind: story
status: draft
title: A scalar read by a reduction or a loss runs on Vulkan
scope:
- confidence: inferred
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
revision: 4
---
# A scalar read by a reduction or a loss runs on Vulkan

## Why

Review of 0.5.0 (finding C2), still open at main 496e98e. The scheduler places a folded scalar as
an immediate operand (`crates/modelspec-core/src/schedule.rs:364-366`), and the SPIR-V emitter
refuses an immediate operand on the kernels that take only buffer operands
(`crates/modelspec-runtime/src/spirv/emit.rs:276-292`). A program whose reduction or loss reads a
literal compiles on CPU and fails on `--backend vulkan`.

Outcome: a reduction or a loss that reads a scalar trains on Vulkan with the CPU's loss in the
comparison class DESIGN.md § 5.4 gives it, and a plan that would hand such a kernel an immediate is
refused by the verifier rather than the emitter.

## Acceptance

- **scalar-sum-trains-on-vulkan**: a program with `loss = add(mse(..), sum(2.0))` trains on
  `--backend vulkan`, and each per-step loss passes against the CPU run's within § 9 gate 5, and bit
  for bit where § 5.4 puts it in the bit-for-bit class.
- **scalar-mse-trains-on-vulkan**: a program with `mse(1.0, 2.0)` in its loss trains on
  `--backend vulkan`, each per-step loss compared with the CPU's the same way.
- **verifier-refuses-immediate-on-reduce-and-loss**: `verify_plan` refuses a hand-built plan that
  gives a reduce or loss kernel an `Immediate` operand, naming the kernel.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/verify.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`

## Out of scope

- Immediate operands on elementwise kernels, which already work.
- Any change to the CPU interpreter.
