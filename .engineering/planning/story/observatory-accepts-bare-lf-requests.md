---
format: aep.planning-md/3
id: story:observatory-accepts-bare-lf-requests
kind: story
status: draft
title: The observatory server answers a bare-LF request
scope:
- confidence: cited
  path: crates/modelspec-observatory/src/server.rs
revision: 2
---
# The observatory server answers a bare-LF request

## Why

Review of 0.5.0 (finding O4), still open at main 496e98e. The request reader stops only at
`\r\n\r\n` (`crates/modelspec-observatory/src/server.rs:233`). A client that ends its head with
`\n\n` waits until the head limit or the socket closes.

Outcome: a request head ended by a blank line in either form gets an answer.

## Acceptance

- **bare-lf-answered**: a request whose head ends `\n\n` is answered within 1 s.
- **crlf-unchanged**: a `\r\n\r\n` request is answered as before.
- `task ci` exits 0.

## Scope

- `crates/modelspec-observatory/src/server.rs`

## Out of scope

- Any other HTTP parsing change.
