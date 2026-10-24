---
format: aep.planning-md/3
id: story:cpu-prepared-training-step
kind: story
status: implemented
title: 'A prepared CPU training step: verify once, free intermediates, move outputs'
relations:
- serves: vision:modelspec
scope:
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-runtime/benches/mnist_shape.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/optimizer.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_prepared_plan.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_prepared_step.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_prepared_step_2.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/prepared_run.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/prepared_step.rs
revision: 10
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T08:09:08Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "proposed", to: "active", at: "2026-10-24T08:09:08Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "active", to: "implemented", at: "2026-10-24T09:37:29Z", actor: "human:timo", revision: 10, decided_on: {"recorded":{"test_result":1,"review_outcome":4,"verification":1}}}
---
# A prepared CPU training step: verify once, free intermediates, move outputs

## Why

Review of 0.7.1 (findings 4 and 7) and Review of 0.5.0 (findings P2 and P4), at main 496e98e. Every
CPU step verifies the step (`crates/modelspec-runtime/src/train/optimizer.rs:126`,
`:137-142`), the program (`crates/modelspec-core/src/ir/verify.rs:414`), and the program again in the
interpreter (`crates/modelspec-runtime/src/cpu/interpreter.rs:138`). Every step also rebuilds the
same tables (P4): `effects_in_order` and the `writes` table
(`crates/modelspec-runtime/src/train/optimizer.rs:135-141`) and `in_place_transposes`
(`crates/modelspec-runtime/src/cpu/interpreter.rs:141-144`, `:166-184`). The interpreter keeps every
value in a `BTreeMap` until the end (`interpreter.rs:143-161`), allocates a new output for every op
(`interpreter.rs:155-157`, finding 7), and copies its outputs out (`interpreter.rs:103-117`; the
optimizer part was fixed in 6272b74).

Outcome: a CPU run verifies its program once and builds `effects_in_order`, the `writes` table and
`in_place_transposes` once per run; during a step it frees each intermediate after its last
consumer, reuses per-op output storage, and moves outputs instead of copying them.

## Acceptance

- **verify-once-per-run**: a test counts one call of each verifier per run, not per step.
- **tables-once-per-run**: a test counts one build of `effects_in_order`, of the `writes` table and
  of `in_place_transposes` per run, not per step.
- **intermediates-freed**: the interpreter exposes to tests a counter of live values. During a step
  of the MNIST training program its peak equals the largest number of values live at once by
  last-consumer analysis of that program, and is below the program's node count (today every value
  stays live, so the peak is the node count). `evaluate_all` still keeps every value.
- **storage-reused**: across two consecutive steps, a counting allocator in the test sees no more
  allocations in the second step than the number of outputs the step returns.
- **outputs-moved**: outputs leave the interpreter without a copy (test by buffer identity).
- **mnist-step-time**: the MNIST CPU benchmark's step time is measured before the change and after
  it, interleaved on one pinned core, and both are reported with the command, the round count and
  the host load; after is not above before at the median by more than 2 %, and the difference is reported as a
  number. Amended 2026-10-24 by the wave coordinator: 101 interleaved rounds gave 0.383 ms before
  and 0.384 ms after at the median with the AVX2 matmul kernel at 75–79 % of a step, so the
  original "after is below before" is inside the host's noise; the structural criteria above
  are what this story changes, and the step-time gain is measured for the wave as a whole at the
  release benchmark rerun.
- **same-bits**: outputs and the per-step loss are bit-identical to before.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/verify.rs`
- `crates/modelspec-runtime/benches/mnist_shape.rs`
- `crates/modelspec-runtime/src/cpu/interpreter.rs`
- `crates/modelspec-runtime/src/train/optimizer.rs`
- `crates/modelspec-runtime/src/train/runner.rs`
- `crates/modelspec-runtime/tests/adversary_prepared_plan.rs`
- `crates/modelspec-runtime/tests/adversary_prepared_step_2.rs`
- `crates/modelspec-runtime/tests/adversary_prepared_step.rs`
- `crates/modelspec-runtime/tests/prepared_run.rs`
- `crates/modelspec-runtime/tests/prepared_step.rs`

## Out of scope

- Batch input allocation, which story:dense-step-reuses-batch-buffers holds.
- The finite check, which story:cpu-finite-check-updated-params-only holds.
