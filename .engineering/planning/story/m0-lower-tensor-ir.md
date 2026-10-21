---
format: aep.planning-md/3
id: story:m0-lower-tensor-ir
kind: story
status: active
title: Lower Typed HIR to Tensor IR and modelspec ir --stage tensor
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
- depends_on: story:m0-check-types-shapes
- depends_on: story:m0-tensor-ir-core
scope:
- confidence: cited
  path: crates/modelspec-cli/src/ir.rs
- confidence: cited
  path: crates/modelspec-cli/tests/golden/linear_regression.tensor.ir
- confidence: cited
  path: crates/modelspec-cli/tests/ir_tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/compile.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/passes/dce.rs
revision: 10
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 9}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 10}
---
# Lower Typed HIR to Tensor IR and `modelspec ir --stage tensor`

## Context

Joins the front end (Typed HIR) to the semantic core (Tensor IR): after this story a `.mspec` file
becomes a verified `TensorProgram`. Implements DESIGN.md § 3 (tensor SSA graph, DCE), § 3.3 pass 2
("lower functions/layers to tensor graph") and the dead-code part of pass 3, § 2.1 (`fn` inlined,
`param` to a parameter handle plus initialization spec), § 2.3 (canonical parameter paths), and the
`ir --stage tensor` verb of DESIGN.md § 8 ("readable tensor IR with stable IDs").

## Lands in

- `crates/modelspec-core/src/lower.rs`: lowers the `train` block loss with the model `forward`
  inlined: `input` and `target` declarations become `Input` nodes, each `param` a `ParamRead` of a
  table entry keyed by its canonical path with its initializer, the loss the single output; spans
  carried to `TensorNode.source_span`. Returns `LoweredProgram { program: TensorProgram,
  prediction: ValueId }`, where `prediction` is the value of the inlined `forward` call (the loss
  depends on it, so DCE keeps it).
- `crates/modelspec-core/src/passes/dce.rs`: removes nodes that reach no output.
- `crates/modelspec-core/src/compile.rs`: the pipeline driver in two public steps,
  `compile_hir(path, source) -> Result<Hir, Vec<Diagnostic>>` (parse, resolve, check) and
  `compile_tensor(path, source) -> Result<LoweredProgram, Vec<Diagnostic>>` (`compile_hir`, then
  lower, DCE, verify). A verifier failure after lowering is a compiler bug and is reported as such,
  not as a user diagnostic (DESIGN.md § 8.1).
- `crates/modelspec-cli/src/ir.rs`: `modelspec ir <file> --stage tensor` prints the program with
  `ir/print.rs`; diagnostics as in `check`. Only the `tensor` stage exists after this story.
- Golden: `crates/modelspec-cli/tests/golden/linear_regression.tensor.ir`;
  test `crates/modelspec-cli/tests/ir_tensor.rs`.

## Out of scope

- `--stage training`, autodiff and SGD (stories `m0-reverse-mode-autodiff`, `m0-cpu-sgd-training`).
- Canonicalization, constant propagation and algebraic simplification (DESIGN.md § 3.3 pass 3): no
  M0 example contains a foldable constant, and § 3.3 allows only IEEE-safe rewrites; deferred.
- Modules, layers and generics (DESIGN.md § 11 iteration 3).
- Edits to `ops.rs`, `ir/verify.rs`, `ir/training.rs` or `autodiff.rs`: story
  `m0-reverse-mode-autodiff` may run beside this one and owns them.

## Domain relations

None. Tensor IR is a compiler construct, not an ESS-typed contract (AGENTS.md).

## Acceptance

`cargo test -p modelspec-cli --test ir_tensor` passes, asserting that
`modelspec ir examples/linear_regression.mspec --stage tensor` exits 0, prints exactly the
checked-in golden (a `ParamRead` of `LinearReg.w` feeding a `MatMul` of type
`Tensor<f32, [8, 1]>` feeding `Mse`), and prints identical bytes on a second run.
