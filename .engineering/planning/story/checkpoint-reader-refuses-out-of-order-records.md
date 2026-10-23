---
format: aep.planning-md/3
id: story:checkpoint-reader-refuses-out-of-order-records
kind: story
status: draft
title: Checkpoint reader refuses records out of path order
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/train/checkpoint.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/checkpoint.rs
revision: 3
---
# Checkpoint reader refuses records out of path order

## Why

Review of 0.5.0 (finding C6), still open at main 496e98e. The checkpoint reader checks the manifest's
parameter records for duplicates only (`crates/modelspec-runtime/src/train/checkpoint.rs:823-847`).
The writer emits them in path order, so a manifest out of that order was not written by ModelSpec,
and reading it gives a second byte form for the same checkpoint.

Outcome: one checkpoint has one accepted byte form; records out of path order are refused.

## Acceptance

- **swapped-records-refused**: a manifest with two parameter records swapped fails `read()` with an
  `E-CKPT` code that names the order and the two paths.
- **written-checkpoint-still-reads**: every checkpoint the writer produces in the test suite still
  reads back.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/train/checkpoint.rs`
- `crates/modelspec-runtime/tests/checkpoint.rs`

## Out of scope

- The size and missing-parameter checks, which story:checkpoint-size-checks-never-wrap holds.
