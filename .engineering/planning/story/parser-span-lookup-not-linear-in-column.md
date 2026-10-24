---
format: aep.planning-md/3
id: story:parser-span-lookup-not-linear-in-column
kind: story
status: implemented
title: Span lookup is not linear in the column
relations:
- serves: vision:modelspec
scope:
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/tests/span_lookup.rs
- confidence: cited
  path: crates/modelspec-core/tests/span_lookup_adversary.rs
- confidence: cited
  path: crates/modelspec-core/tests/span_lookup_counter_adversary.rs
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T08:09:09Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-10-24T08:09:09Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-10-24T09:37:29Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"test_result":1,"review_outcome":2,"verification":1}}}
---
# Span lookup is not linear in the column

## Why

Review of 0.5.0 (finding P1), still open at main 496e98e. `span_at` counts the characters from the
start of the line to every offset it is asked about
(`crates/modelspec-core/src/syntax/parser.rs:75-84`), so a long line costs time quadratic in its
length.

Outcome: checking a program costs the same whether its statements share a line or not.

## Acceptance

- **long-line-check-time**: `modelspec check` of 20 000 `let`s on one line takes at most 1.2x the time
  of the same 20 000 one per line (median of 3).
- **columns-unchanged**: every diagnostic's line and column in the test suite is unchanged,
  including lines with multi-byte characters.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/syntax/parser.rs`
- `crates/modelspec-core/tests/span_lookup_adversary.rs`
- `crates/modelspec-core/tests/span_lookup_counter_adversary.rs`
- `crates/modelspec-core/tests/span_lookup.rs`

## Out of scope

- Other parser performance.
