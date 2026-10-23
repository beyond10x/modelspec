---
format: aep.planning-md/3
id: story:float-literals-round-once
kind: story
status: draft
title: Float literals round once to f32
scope:
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
revision: 2
---
# Float literals round once to f32

## Why

Review of 0.5.0 (finding C3), still open at main 496e98e. The parser reads a float literal as
`f64` (`crates/modelspec-core/src/syntax/parser.rs:696-697`) and the type layer narrows it to `f32`
later (`crates/modelspec-core/src/types.rs:425`, `:560`). Two roundings can give a different `f32`
than one: a literal just above a halfway point rounds to the halfway `f64`, then to even.

Outcome: every `f32` literal has the bits `str::parse::<f32>` gives its source text.

## Acceptance

- **literal-lr-rounds-once**: `lr = 1.0000000596046447753906250001` compiles to `1.0000001`, the
  same bits as `"1.0000000596046447753906250001".parse::<f32>()`.
- **literal-operand-rounds-once**: the same literal as a `mul` operand gives the same bits.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/syntax/parser.rs`
- `crates/modelspec-core/src/types.rs`

## Out of scope

- Literals of other dtypes; the language has only `f32` tensors today.
