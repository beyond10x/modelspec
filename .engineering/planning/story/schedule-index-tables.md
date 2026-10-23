---
format: aep.planning-md/3
id: story:schedule-index-tables
kind: story
status: draft
title: Schedule and verify through index tables
scope:
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: inferred
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
revision: 3
---
# Schedule and verify through index tables

## Why

Review of 0.7.1 (finding 8) and Review of 0.5.0 (finding P8), at main 496e98e. `TensorProgram::node`
finds a node by linear search (`crates/modelspec-core/src/ir/tensor.rs:219-220`), and the scheduler
calls it and similar scans per node (`crates/modelspec-core/src/schedule.rs:173`, `:304`, `:571`,
`:597`, `:605`), so scheduling is quadratic in program size.

Outcome: node and consumer lookups are table lookups, and scheduling a large program takes linear
time.

## Acceptance

- **ten-thousand-nodes**: schedule plus verify of a generated 10 000-node program completes in under
  1 s.
- **same-plans**: every plan the test suite schedules is identical to before.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/tensor.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/verify.rs`

## Out of scope

- Parser performance, which story:parser-span-lookup-not-linear-in-column holds.
