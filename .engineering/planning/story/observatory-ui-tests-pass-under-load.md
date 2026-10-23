---
format: aep.planning-md/3
id: story:observatory-ui-tests-pass-under-load
kind: story
status: draft
title: Observatory UI browser tests pass under load
revision: 1
---
# Observatory UI browser tests pass under load

## Why

`observatory_ui_adversary` and `observatory_ui_live_*` (`crates/modelspec-cli/tests/`) failed in
5 or more `task ci` runs of the benchmark wave on 2026-10-23 at load averages between 13 and 69,
and passed on rerun with no change. Each failure cost a full gate rerun.

## Acceptance

- The failing waits are found and named; each waits on the state it checks (a DOM condition or
  an event count), with no fixed sleep, and its timeout is stated once.
- The UI tests pass 10 runs in a row while 16 other build jobs run on the host.
- `task ci` exits 0.
