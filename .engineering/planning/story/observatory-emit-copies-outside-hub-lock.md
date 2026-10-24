---
format: aep.planning-md/3
id: story:observatory-emit-copies-outside-hub-lock
kind: story
status: draft
title: Slow clients do not stall emit
scope:
- confidence: cited
  path: crates/modelspec-observatory/src/server.rs
revision: 3
---
# Slow clients do not stall emit

## Why

Review of 0.5.0 (finding O3), still open at main 496e98e. A streaming client concatenates the frames
it has not sent while holding the hub lock (`crates/modelspec-observatory/src/server.rs:300`). The
training loop's emit takes the same lock, so many or slow clients delay training steps.

The mechanism of the fix: a client takes the frame range under the lock and copies and writes it
after releasing the lock.

Outcome: emit never waits on a client's copy or write.

## Acceptance

- **emit-never-waits**: with one connected client that reads nothing, `emit` of 1,000 frames returns
  within 10 ms.
- `task ci` exits 0.

## Scope

- `crates/modelspec-observatory/src/server.rs`

## Out of scope

- A bound on the number of clients or on frame memory.
