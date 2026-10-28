---
format: aep.planning-md/3
id: story:decider-model-function-calls
kind: story
status: draft
title: Calls between model functions, inlined
summary: A model function calls another by name, parameters passed as arguments, inlined at lowering.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/check_model_calls.rs
- confidence: cited
  path: crates/modelspec-core/src/hir.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/gradients_model_calls.rs
- confidence: cited
  path: website
revision: 4
---
# Calls between model functions, inlined

`depends_on: none`

## Outcome

A model function may call another model function of the same model by its bare name, passing
tensors, parameters included, as arguments: `let h1 = encoder_layer(h0, e0_wq, e0_wk, …);`. The
call is inlined when the program is lowered, so the Tensor IR, the training step and the kernels
are those of the hand-inlined program, and each parameter passed keeps its identity, its path and
its gradient. A function that calls itself, directly or through others, is refused at the call.
The `train` block's rule is unchanged: its loss calls `model.forward` exactly once (E-TYPE-017).

## Context

- Gap row 11. Decider design § 5.2: 4 identical event-encoder layers and 2 context-encoder
  layers. The draft file writes 96 layer parameters and 42 layer lines by hand.
- `crates/modelspec-core/src/resolve.rs:12-13`: "model functions are reached only through `model.`
  in the train block, so a model function cannot call itself or another." `module` and `layer`
  are E-SYNTAX-004/005 (`crates/modelspec-core/src/syntax/mod.rs:42-45`), deferred to DESIGN.md
  § 11 iteration 3; DESIGN.md § 2.1 lists `fn` as "Inline or call graph specialization".
- Lowering: `crates/modelspec-core/src/lower.rs`; HIR references: `crates/modelspec-core/src/hir.rs`
  (`FnRef`, `ParamRef`, `ArgRef`).

## Spec first

No noun is added to `spec/`: the change is to the language and the compiler.

## Scope

- `crates/modelspec-core/src/resolve.rs`, `types.rs`, `hir.rs`, `lower.rs`.
- Tests (new): `crates/modelspec-cli/tests/check_model_calls.rs` with fixtures under
  `crates/modelspec-cli/tests/fixtures/`, `crates/modelspec-runtime/tests/gradients_model_calls.rs`.
- `DESIGN.md` § 2.1, § 3.3, § 8.1; `crates/modelspec-docs/src/diagnostics.rs`, then `task
  docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

- **calls-inline-identical**: `cargo test -p modelspec-cli --test check_model_calls`: a two-layer
  MLP written with one `fn dense(x, w, b)` called twice prints the same `ir --stage tensor` and
  `ir --stage training` text as the same program written out by hand, and 10 CPU training steps of
  the two print identical `step` lines.
- **calls-refusals**: in the same test, direct recursion, mutual recursion, an unknown function,
  a wrong argument count and an argument of the wrong type are each refused at the call with
  their code; `forward` called from another model function is refused too.
- **calls-gradients**: `cargo test -p modelspec-runtime --test gradients_model_calls`: a parameter
  passed to two calls of one function gets the sum of both uses' gradients, checked against
  central differences within `TOL_F64` in `f64`.
- `task ci` exits 0, including `task docs-check`.

## Out of scope

- `module`, `layer`, generic parameters and stable instance names (DESIGN.md § 11 iteration 3).
- A call graph that is not inlined; returning more than one tensor.
- Calling a model function from `infer --function` other than as the named entry (S3).
