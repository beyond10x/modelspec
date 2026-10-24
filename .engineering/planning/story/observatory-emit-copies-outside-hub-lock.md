---
format: aep.planning-md/3
id: story:observatory-emit-copies-outside-hub-lock
kind: story
status: draft
title: Slow clients do not stall emit
scope:
- confidence: cited
  path: crates/modelspec-observatory/src/server.rs
revision: 4
---
# Slow clients do not stall emit

## Why

Review of 0.5.0 (finding O3), still open at main 496e98e. A streaming client concatenates the frames
it has not sent while holding the hub lock (`crates/modelspec-observatory/src/server.rs:300`). The
training loop's emit takes the same lock, so a client catching up on a long backlog delays training
steps for as long as the copy takes.

A client that reads nothing does not hold the lock while it blocks: it drops the lock
(`server.rs:303`) before `write_all` (`server.rs:304`). So emit with a non-reading client already
returns promptly at HEAD, and only the backlog copy runs under the lock.

The mechanism of the fix: a client takes the frame range under the lock and copies and writes it
after releasing the lock.

Outcome: no frame bytes are copied while the hub lock is held, so emit never waits on a client's copy
or write.

## Acceptance

- **no-copy-under-lock**: a test hook in the hub counts the bytes copied while the hub lock is held.
  A client that connects after 10,000 frames have been emitted and catches up on that backlog leaves
  the count at 0. At HEAD the count is the byte length of the backlog.
- **emit-never-waits** (regression guard, passes at HEAD): with one connected client that reads
  nothing, `emit` of 1,000 frames returns within 10 ms.
- `task ci` exits 0.

## Scope

- `crates/modelspec-observatory/src/server.rs`

## Out of scope

- A bound on the number of clients or on frame memory.
