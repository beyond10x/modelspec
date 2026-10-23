---
format: aep.planning-md/3
id: story:unclosed-train-reports-missing-brace
kind: story
status: draft
title: An unclosed `train` reports a missing `}`
scope:
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/tests/syntax.rs
revision: 2
---
# An unclosed `train` reports a missing `}`

## Why

Review of 0.5.0 (finding C9), still open at main 496e98e. When a `train` block is not closed, the
parser treats the `model` keyword as a setting position (`crates/modelspec-core/src/syntax/parser.rs:527`)
and reports an unexpected token, not the missing brace.

Outcome: the first diagnostic for an unclosed `train` names the missing `}`.

## Acceptance

- **unclosed-train-diagnostic**: `train {` followed by `model M {` gives a diagnostic that says a `}`
  is missing, at the `model` token.
- **syntax-test-updated**: `crates/modelspec-core/tests/syntax.rs:1005`
  (`unclosed_train_stops_at_every_model_item_head`) asserts the new message.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/syntax/parser.rs`
- `crates/modelspec-core/tests/syntax.rs`

## Out of scope

- Recovery after the diagnostic; the parser still reports only the first error.
