---
format: aep.planning-md/3
id: story:schedule-index-tables
kind: story
status: draft
title: Schedule and verify through index tables
scope:
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
revision: 6
---
# Schedule and verify through index tables

## Why

Review of 0.7.1 (finding 8) and Review of 0.5.0 (finding P8), at main 496e98e. Finding 8 names four
linear lookups, and P8 a quadratic check in the verifier:

- node lookup: `TensorProgram::node` searches linearly
  (`crates/modelspec-core/src/ir/tensor.rs:219-220`), and the scheduler's producer scans for fusion
  do the same (`crates/modelspec-core/src/schedule.rs:173`, `:224`);
- kernel dedup: each kernel is compared with every kernel kept so far
  (`schedule.rs:569-573`);
- parameter and view bindings: found by linear search (`schedule.rs:304`, `:597`, `:605`);
- hazard pairs: the verifier's step-ordering closure is quadratic in the step count
  (`crates/modelspec-core/src/ir/verify.rs:1313-1347`).

Together they make scheduling and verifying quadratic in program size.

Outcome: node, producer, kernel and binding lookups are table lookups, the hazard check is no longer
quadratic in the step count, and scheduling plus verifying a large program grows linearly with it.

## Acceptance

- **linear-growth**: schedule plus verify of a generated 20 000-node program takes at most 2.5× the
  time of the same generator's 10 000-node program (median of 5 runs each, same process).
- **same-plans**: every plan the test suite schedules is identical to before.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/tensor.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/verify.rs`

## Out of scope

- Parser performance, which story:parser-span-lookup-not-linear-in-column holds.
