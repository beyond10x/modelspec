---
format: aep.planning-md/3
id: story:checkpoint-size-checks-never-wrap
kind: story
status: draft
title: Checkpoint size and parameter checks never wrap or drop
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/train/checkpoint.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/checkpoint.rs
revision: 3
---
# Checkpoint size and parameter checks never wrap or drop

## Why

Review of 0.5.0 (finding C8), still open at main 496e98e. `of_state` skips a declared parameter that
has no value (`values.get(id)?` at `crates/modelspec-runtime/src/train/checkpoint.rs:377-386`), and
the reader converts `u64` sizes with `as usize` (`:685`, `:776`), which truncates on a 32-bit host.

Outcome: a checkpoint is never written without a declared parameter, and no size conversion
truncates.

## Acceptance

- **missing-param-errors**: `of_state` given a state without one declared parameter returns an error
  naming its path instead of a checkpoint without it.
- **sizes-use-try-from**: every `u64` size in the reader goes through `usize::try_from`, and a size
  above `usize::MAX` is refused with an `E-CKPT` code (host test with a crafted header).
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/train/checkpoint.rs`
- `crates/modelspec-runtime/tests/checkpoint.rs`

## Out of scope

- Record order, which story:checkpoint-reader-refuses-out-of-order-records holds.
