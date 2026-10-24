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
revision: 4
---
# Checkpoint size and parameter checks never wrap or drop

## Why

Review of 0.5.0 (finding C8), still open at main 496e98e. `of_state` skips a declared parameter that
has no value (`values.get(id)?` at `crates/modelspec-runtime/src/train/checkpoint.rs:377-386`), and
the reader converts `u64` sizes with `as usize` (`:685`, `:776`), which truncates on a 32-bit host.

The mechanism of the fix is `usize::try_from` on every `u64` size the reader converts. A `u64` above
`usize::MAX` cannot occur on a 64-bit host, so that refusal is stated as the mechanism and not
tested; what is tested on 64-bit is that a declared size larger than the file is refused.

Outcome: a checkpoint is never written without a declared parameter, and no size conversion
truncates.

## Acceptance

- **missing-param-errors**: `of_state` given a state without one declared parameter returns an error
  naming its path instead of a checkpoint without it.
- **declared-size-past-file**: on a 64-bit host, a checkpoint whose header declares a manifest size,
  and one whose manifest declares a tensor extent, larger than the file is refused with `E-CKPT-002`
  (`crates/modelspec-runtime/src/train/checkpoint.rs:41-46`), before any buffer of that size is
  allocated.
- **sizes-use-try-from** (mechanism): every `u64` size the reader converts goes through
  `usize::try_from`, and a failure maps to `E-CKPT-002`; no `as usize` on a size read from the file
  remains.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/train/checkpoint.rs`
- `crates/modelspec-runtime/tests/checkpoint.rs`

## Out of scope

- Record order, which story:checkpoint-reader-refuses-out-of-order-records holds.
