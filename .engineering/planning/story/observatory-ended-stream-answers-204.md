---
format: aep.planning-md/3
id: story:observatory-ended-stream-answers-204
kind: story
status: draft
title: An ended stream answers 204 to a reconnect
scope:
- confidence: cited
  path: crates/modelspec-observatory/src/server.rs
revision: 2
---
# An ended stream answers 204 to a reconnect

## Why

Review of 0.5.0 (finding O2), still open at main 496e98e. After a run's event stream has ended, a
client that reconnects with `Last-Event-ID` gets `200` and an empty stream
(`crates/modelspec-observatory/src/server.rs:283-305`), so an `EventSource` reconnects forever.

Outcome: a reconnect to an ended stream is told it is over.

## Acceptance

- **ended-reconnect-204**: a request with `Last-Event-ID: <last id>` after the end of the stream gets
  `204 No Content`.
- **mid-run-reconnect-unchanged**: a reconnect before the end still resumes after the given id.
- `task ci` exits 0.

## Scope

- `crates/modelspec-observatory/src/server.rs`

## Out of scope

- The UI's handling of the end of a run.
