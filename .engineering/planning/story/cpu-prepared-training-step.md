---
format: aep.planning-md/3
id: story:cpu-prepared-training-step
kind: story
status: draft
title: 'A prepared CPU training step: verify once, free intermediates, move outputs'
scope:
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/optimizer.rs
revision: 2
---
# A prepared CPU training step: verify once, free intermediates, move outputs

## Why

Review of 0.7.1 (findings 4 and 7) and Review of 0.5.0 (findings P2 and P4), at main 496e98e. Every
CPU step verifies the step (`crates/modelspec-runtime/src/train/optimizer.rs:126`,
`:137-142`), the program (`crates/modelspec-core/src/ir/verify.rs:414`), and the program again in the
interpreter (`crates/modelspec-runtime/src/cpu/interpreter.rs:138`). The interpreter keeps every
value in a `BTreeMap` until the end (`interpreter.rs:143-161`) and copies its outputs out
(`interpreter.rs:103-117`; the optimizer part was fixed in 6272b74).

Outcome: a CPU run verifies its program once, frees each intermediate after its last consumer, and
moves outputs instead of copying them.

## Acceptance

- **verify-once-per-run**: a test counts one call of each verifier per run, not per step.
- **intermediates-freed**: during a step, a value is dropped after its last consumer; `evaluate_all`
  still keeps every value.
- **outputs-moved**: outputs leave the interpreter without a copy (test by buffer identity).
- **mnist-step-time**: the MNIST CPU benchmark's step time is reported before and after.
- **same-bits**: outputs and the per-step loss are bit-identical to before.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/cpu/interpreter.rs`
- `crates/modelspec-runtime/src/train/optimizer.rs`
- `crates/modelspec-core/src/ir/verify.rs`

## Out of scope

- Batch input allocation, which story:dense-step-reuses-batch-buffers holds.
- The finite check, which story:cpu-finite-check-updated-params-only holds.
