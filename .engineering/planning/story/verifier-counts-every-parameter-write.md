---
format: aep.planning-md/3
id: story:verifier-counts-every-parameter-write
kind: story
status: draft
title: Any step-plan write to a parameter buffer counts as an update
scope:
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
revision: 2
---
# Any step-plan write to a parameter buffer counts as an update

## Why

Review of 0.5.0 (finding C7), still open at main 496e98e. The plan verifier's check that a parameter
is not read after it is updated keys on `SgdUpdate` dispatches only
(`crates/modelspec-core/src/ir/verify.rs:1385-1441`). A plan whose other dispatch writes a parameter
buffer passes, and a later read in the same recording sees the new value.

Outcome: the verifier treats every dispatch that writes a parameter's buffer as an update of it.

## Acceptance

- **elementwise-write-to-param-refused**: a hand-built plan whose `Elementwise` dispatch writes a
  parameter buffer that a later dispatch in the same recording reads fails `verify_plan`, naming
  both dispatches.
- **scheduled-plans-still-verify**: every plan `schedule` produces in the test suite still passes.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/verify.rs`

## Out of scope

- Changing which kernels the scheduler lets write parameters.
