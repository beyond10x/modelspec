---
format: aep.planning-md/3
id: story:serve-ends-without-signal-handler
kind: story
status: draft
title: '`--serve` ends cleanly when no signal handler can be installed'
scope:
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
revision: 2
---
# `--serve` ends cleanly when no signal handler can be installed

## Why

Review of 0.5.0 (finding O5), still open at main 496e98e. When `set_handler` fails (a second
in-process dispatch already installed one), `train --serve` falls through to `server.wait()`
(`crates/modelspec-cli/src/train.rs:548-555`, `:565-573`), which never returns.

Outcome: `train --serve` returns whether or not it could install its handler.

## Acceptance

- **second-serve-returns**: a second in-process dispatch of `train --serve` returns its exit code
  instead of hanging (test with a timeout).
- `task ci` exits 0.

## Scope

- `crates/modelspec-cli/src/train.rs`

## Out of scope

- Signal handling of the first dispatch.
