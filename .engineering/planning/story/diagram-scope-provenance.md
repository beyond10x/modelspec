---
format: aep.planning-md/3
id: story:diagram-scope-provenance
kind: story
status: active
title: 'Scope provenance in the Tensor IR: every node knows where it came from'
summary: A scope table in TensorProgram stamped by lowering (train block, function inlining with instance and call site), autodiff (adjoint of a forward node) and optimizer attach (update of a parameter); ir output unchanged.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/training.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/passes/dce.rs
- confidence: inferred
  path: crates/modelspec-core/tests/scopes.rs
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T19:01:27Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "proposed", to: "active", at: "2026-10-28T19:01:27Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"review_outcome":2}}}
---
# Scope provenance in the Tensor IR: every node knows where it came from

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.2. Zoom needs every node's scope path: the train block, each
inlined function and its call site, the gradient rule of one forward node, the update of one
parameter. Lowering knows the inlining stack and throws it away: `Lowering.inlining` holds
`(FnRef, instance)` while a call is inlined (`crates/modelspec-core/src/lower.rs:234`, pushed and
popped in `fn inline`, `:367`), and the instance count already names dropout sites `<fn>@<c>`
(`:322`). A `TensorNode` keeps only its `source_span` (`crates/modelspec-core/src/ir/tensor.rs:311`).
Spans cannot tell two inlinings of one function apart, so the scope has to be recorded where the
inlining happens.

## What it adds

- `TensorProgram.scopes: ScopeTable` (`crates/modelspec-core/src/ir/tensor.rs`): a list of scope
  entries `(parent, ScopeSegment)` and the scope of every `ValueId`, beside the nodes as
  `TrainingProgram.subgraphs` sits beside them. `ScopeSegment` is `Program(name)`,
  `Function { name, instance, call_span }`, `Adjoint(ValueId)`, `Update(param path)`, `Module` and
  `Layer` (typed, emitted by nothing yet).
- `TensorProgramBuilder::enter_scope` and `exit_scope`; every node the builder adds is stamped with
  the current scope, and scopes nest as deep as they are entered.
- `lower`: enters `Program` first and `Function` in `fn inline` with the instance it already counts.
- `autodiff::differentiate` and `differentiate_inputs`: every node a rule emits is stamped
  `Adjoint(%k)` under the forward node's scope, the `gradient`-tagged nodes and the constants and
  transposes the IR tags `forward` alike. The seed `dL/dL` stays in the Program scope.
- `attach_sgd`, `attach_adamw`, `attach_muon`: each parameter's update nodes stamped
  `Update(<path>)`. A value created before any one parameter's update and shared by all of them
  (the learning-rate `Scalar`, the AdamW step inputs) stays in the Program scope (DESIGN.md § 13.2).
- DCE and every pass: keep the entry of each surviving node and drop the rest.
- `verify`: refuses a node with no scope and a scope whose parent chain does not reach the
  Program scope (a new `V-` rule).
- `modelspec ir` prints nothing new.

## Order on shared files

`story:decider-model-function-calls` (draft, `epic:decider-training-support`) also edits
`lower.rs` (`fn inline`) and `resolve.rs`. Whichever merges second rebases onto the other. Nested
Function scopes then follow from the builder's scope stack, which acceptance 2 checks today at the
builder; the end-to-end case of a model function calling another is outside this story (see Out of
scope).

## Scope

- `crates/modelspec-core/src/ir/tensor.rs`, `src/lower.rs`, `src/autodiff.rs`,
  `src/ir/training.rs`, `src/ir/verify.rs`, `src/passes/dce.rs` (cited).
- `crates/modelspec-core/tests/scopes.rs` (new).
- `DESIGN.md` § 3.2 (one paragraph naming the table), `CHANGELOG.md` one line.

## Acceptance

Tests in `crates/modelspec-core/tests/scopes.rs` (`cargo test -p modelspec-core`, in `task ci`):

1. `scope.lowering-stamps-functions`: for the program of the dropout-site test in `lower.rs` (a
   `forward` and a function `extra` inlined twice from the loss), every forward node's scope key is
   `<Train>`, `<Train>/forward`, `<Train>/extra` or `<Train>/extra@1`, the `@1` instance's call span
   is the second call's, and each dropout token named `<Model>.extra@1.dropout#k` lies in
   `<Train>/extra@1`.
2. `scope.builder-nests`: a `TensorProgramBuilder` that enters `Program(T)`, `Function(a, 0)` and
   `Function(b, 1)`, pushes a node, exits twice and pushes another stamps the first `T/a/b@1` and the
   second `T`, and the table's parent chain of `T/a/b@1` is `T/a`, then `T`.
3. `scope.adjoints-and-updates`: in the training step of `two_layer_relu.mspec`
   (`crates/modelspec-docs/models/two_layer_relu.mspec`, `modelspec ir --stage training`):
   - every `gradient` node's innermost segment is `Adjoint(%k)` with `%k` a forward node;
   - so is that of each forward-tagged rule value, `%9 Sub`, `%10 Scalar`, `%13 Transpose`,
     `%15 Transpose` and `%18 Transpose`;
   - `%21` and `%22` lie in `Update(@TwoLayer.w1)` and `%23` and `%24` in `Update(@TwoLayer.w2)`;
   - the shared learning rate `%20` and the seed `%8` lie in the Program scope;
   - every node has an entry.
4. `scope.survives-dce`: after `passes::dce` every surviving node keeps its scope, and no entry is
   left for a removed node.
5. `scope.verifier-refuses-missing`: a program with one node's entry removed is refused by `verify`
   with the new rule's code.
6. `scope.ir-text-unchanged`: `cargo test -p modelspec-cli` passes with the golden files under
   `crates/modelspec-cli/tests/golden/` unchanged, and `ir_hash` of `examples/bert_tiny.mspec`
   equals the value before the change (recorded in the test).

## Out of scope

- Drawing anything (`story:diagram-zoom` reads the table).
- A source fixture in which one model function calls another: the language refuses it until
  `story:decider-model-function-calls` lands, and that story's tests are where the case is added.
- Module and layer instances (DESIGN.md § 11 iteration 3).
