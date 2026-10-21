---
format: aep.planning-md/3
id: story:m0-parse-subset
kind: story
status: implemented
title: Parse the M0 language subset
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
- depends_on: story:m0-workspace-foundation
scope:
- confidence: cited
  path: crates/modelspec-core/src/syntax/ast.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/lexer.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/mod.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/tests/syntax.rs
- confidence: cited
  path: crates/modelspec-core/tests/syntax_adversary.rs
- confidence: cited
  path: tests/language/syntax
revision: 12
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:26Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:26Z", actor: "human:timo", revision: 10, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "active", to: "implemented", at: "2026-10-21T14:53:40Z", actor: "human:timo", revision: 12, decided_on: {"recorded":{"test_result":1,"review_outcome":2,"verification":1}}}
---
# Parse the M0 language subset

## Context

First stage of the pipeline in DESIGN.md § 3: `.mspec` source to Source AST, for the restricted M0
subset of DESIGN.md § 2 (one model, static dimensions, `f32`, `param`, local `let`, call syntax for
`matmul`, `add`, `sub`, `mul`, `relu`, `mean`, `mse`, and a `train` block with `sgd`). Implements
DESIGN.md § 3.3 pass 1 (parse) and the parser half of DESIGN.md § 9 gate 1 (round-trip tests).
The target input is `examples/linear_regression.mspec`.

## Lands in

Crate `modelspec-core`, files created empty by the foundation story:

- `src/syntax/lexer.rs`: tokens with `logos` (DESIGN.md § 7), `//` line comments, identifiers,
  integer and float literals, punctuation, keywords `model`, `param`, `fn`, `let`, `return`, `train`,
  `input`, `target`.
- `src/syntax/ast.rs`: Source AST with a `SourceSpan` on every node.
- `src/syntax/parser.rs`: hand-written recursive descent. Grammar: one `model Name { ... }` holding
  `param name: Type = init();` and `fn name(args) -> Type { let ...; return expr; }`; one
  `train Name { ... }` holding `model = Name;`, `input`/`target` declarations, `loss = expr;`,
  `optimizer = sgd(lr = <float>);`, `steps = <int>;`, `seed = <int>;`, `device = cpu;` or
  `device = vulkan(<int>);`. Expressions: identifiers, literals, calls with positional and named
  arguments, and member calls such as `model.forward(x)`. Types: `Tensor<f32, [d0, d1, ...]>` with
  integer literal dimensions; `[]` is rank 0.
- `src/syntax/mod.rs`: `parse(file, source) -> Result<Ast, Vec<Diagnostic>>` and a canonical AST
  printer used for round-trip tests.
- Syntax diagnostics `E-SYNTAX-0nn`, declared in `src/syntax/`, rendered through `diag.rs`. A
  construct outside the M0 subset (`module`, `layer`, generic `<B: dim>` parameters, infix operators)
  gets its own `E-SYNTAX` code naming the construct, not a generic "unexpected token".
- Tests: `crates/modelspec-core/tests/syntax.rs`; negative fixtures under `tests/language/syntax/`.

## Out of scope

- Any `Cargo.toml` or `Cargo.lock` edit: the foundation story declares `logos` for
  `modelspec-core`.
- Name resolution, type and shape checking, the `modelspec check` verb (story
  `m0-check-types-shapes`).
- Modules, layers, dimension generics, `const`, `repeat`, infix arithmetic: DESIGN.md § 11 puts them
  in iteration 3. `examples/tiny_mlp.mspec` uses them and is not expected to parse in M0.
- Error recovery beyond reporting the first syntax error per item.

## Domain relations

None. The `.mspec` language is not an ESS domain (AGENTS.md; DESIGN.md § 2).

## Acceptance

`cargo test -p modelspec-core --test syntax` passes, including a round-trip of
`examples/linear_regression.mspec` (parse, print, re-parse yields an AST equal up to spans) and one
negative fixture per `E-SYNTAX` code asserting that code and its line and column.
