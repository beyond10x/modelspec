---
format: aep.planning-md/3
id: story:bert-embedding-lookup
kind: story
status: draft
title: Integer token inputs and the embedding lookup, gradient-checked on the CPU
summary: Adds the i32 element type, restricted to ids and labels, and embedding with a deterministic scatter-add gradient on the CPU interpreter, with an f64 finite-difference check.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:observatory-saliency-gradient-x-input
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_i32.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/
- confidence: cited
  path: crates/modelspec-cli/tests/train_i32_synthetic.rs
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/ops.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/ast.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/lexer.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-observatory/
- confidence: cited
  path: crates/modelspec-observatory/schema/
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/tensor.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/common/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients_embedding.rs
- confidence: cited
  path: spec/domains/checkpoint.yaml
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated/
- confidence: cited
  path: website/docs/reference/diagnostics.md
- confidence: cited
  path: website/docs/reference/operators.md
revision: 3
---
# Integer token inputs and the embedding lookup, gradient-checked on the CPU

## Context

`epic:bert-from-scratch` (In scope, Language and operators: "embedding lookup, positional embeddings"). M0 has
one element type: `DType::F32` (`crates/modelspec-core/src/ir/tensor.rs:38`); the parser accepts
only `f32` in a type (`crates/modelspec-core/src/syntax/parser.rs`, module doc grammar,
`type := "Tensor" "<" "f32" ...`), and `modelspec.checkpoint.DType` has the one variant `F32`
(`spec/domains/checkpoint.yaml`). Token ids are integers, so this story adds `i32` and the first
operator that reads it.

This is the first of seven operator stories (embedding, dense and GELU, layer norm and softmax,
attention, cross-entropy, dropout, AdamW). Each adds its operators to the compiler and the CPU
interpreter only, and each carries its own finite-difference check on the CPU interpreter
(DESIGN.md § 9 gates 1-3). GPU kernels are `story:bert-vulkan-kernels-rowwise` and
`story:bert-vulkan-kernels-attention-loss-adamw`, after `epic:m1-vulkan-execution`.

## What this story decides

- `i32` joins `f32` as an element type (DESIGN.md § 2.2 says signed integers come later). An
  `i32` tensor may be declared only by an `input` or `target` of a `train` block or by a function
  parameter, and may be used only as the index operand of `embedding` (and, from
  `story:bert-mlm-cross-entropy`, as labels). Any other use (arithmetic, `matmul`, a `param` of
  type `i32`, a function result of type `i32`) is refused by the checker with one new `E-TYPE`
  code. There is no implicit conversion either way.
- `embedding(table, ids)`: `table: Tensor<f32, [V, H]>`, `ids: Tensor<i32, [N]>`, result
  `Tensor<f32, [N, H]>`, row `n` being row `ids[n]` of `table`. Shapes that do not fit are an
  `E-TYPE` error; an id outside `[0, V)` at run time is a `RuntimeError` of the CPU interpreter,
  exit 1.
- The gradient with respect to `table` is an internal gradient op `EmbeddingGrad` (DESIGN.md § 4,
  as `ReluGrad` is): a scatter-add of the output adjoint into a `[V, H]` zero tensor, rows
  accumulated in ascending `n`, so the result is the same on every run. `ids` has no gradient.
- Positional embeddings are `embedding(pos_table, positions)` over an `i32` input `positions`
  supplied with the batch; there is no separate positional operator.
- `train` on synthetic data (`crates/modelspec-runtime/src/train/data.rs`) refuses a declared `i32`
  input or target with a new `E-DATA-004`, because M0 synthetic inputs are standard normal `f32`
  draws (DESIGN.md § 2.3, stream 2). `story:bert-train-data-binding` supplies integer data.

## Order on shared files

The seven operator stories each edit `crates/modelspec-core/src/ops.rs`,
`crates/modelspec-core/src/resolve.rs` (`OPERATORS`), `crates/modelspec-core/src/autodiff.rs`,
`crates/modelspec-runtime/src/cpu/interpreter.rs`, `DESIGN.md` § 4 and the revision line,
`CHANGELOG.md`, `crates/modelspec-docs/src/operators.rs` with `website/docs/reference/operators.md`,
and the `TensorOperator` variants of `spec/domains/observatory.yaml` (its summary: "The operators
of epic:bert-from-scratch join as the compiler gains them") with their mapping in
`crates/modelspec-observatory/`, the committed observatory JSON Schema and `ui/src/generated/`.
They run in the order embedding-lookup, dense-gelu, layer-norm-softmax, attention,
mlm-cross-entropy, dropout, adamw-warmup, each with a `depends_on` edge to the one before.

This story depends on `story:observatory-saliency-gradient-x-input`, the last observatory story to
edit `spec/domains/observatory.yaml`, `crates/modelspec-observatory/`, `ui/src/generated/` and
`CHANGELOG.md`. Through its edges it also follows `story:observatory-input-adjoint` (`autodiff.rs`,
`DESIGN.md` § 3.3 and § 4) and `story:observatory-activation-statistics` (`runner.rs`,
`optimizer.rs`).

## Domain relations

- `modelspec.checkpoint.DType` (`spec/domains/checkpoint.yaml`) gains `I32` with `size_bytes: 4`.
  `modelspec.artifact.ParameterSpec.dtype`, `modelspec.artifact.DataSpec.dtype`
  (`spec/domains/artifact.yaml`) and `modelspec.observatory.GraphNode.dtype`
  (`spec/domains/observatory.yaml`) are typed by that enum; no `relations:` entry changes.
- A parameter stays `F32`: the checker refuses an `i32` `param`, so every
  `modelspec.artifact.ParameterSpec.dtype` is `F32`. Inferred from the rule above and from
  `crates/modelspec-core/src/ir/tensor.rs:38` (one dtype today); no ESS invariant states it.

## Scope

- `crates/modelspec-core/src/syntax/{lexer,parser,ast}.rs`: `i32` in a type.
- `crates/modelspec-core/src/types.rs`, `resolve.rs`, `ops.rs`: the dtype rule, the new `E-TYPE`
  code, `embedding` in `OPERATORS` with arity, inference and gradient.
- `crates/modelspec-core/src/ir/tensor.rs` (`DType::I32`, `TensorOp::Embedding`,
  `TensorOp::EmbeddingGrad`), `ir/verify.rs`, `ir/print.rs`, `autodiff.rs`, `lower.rs`.
- `crates/modelspec-runtime/src/cpu/{tensor,interpreter}.rs`: `i32` values held exactly, the two
  ops, the out-of-range error.
- `crates/modelspec-runtime/src/train/data.rs`: `E-DATA-004`.
- `crates/modelspec-runtime/tests/common/mod.rs` (new): the finite-difference harness of
  `tests/gradients.rs`, moved with its constants unchanged (`H`, `TOL_F64`, `TOL_F32`,
  `KINK_MARGIN`, `MIN_SCALE`, lines 26-35) and extended with integer inputs that are held fixed
  while `f32` inputs and parameters are perturbed; `tests/gradients.rs` uses it.
  `tests/gradients_embedding.rs` (new).
- `crates/modelspec-cli/tests/fixtures/` (new): `.mspec` fixtures; `tests/check_i32.rs`,
  `tests/train_i32_synthetic.rs` (new).
- `spec/domains/checkpoint.yaml` (`DType`), `spec/domains/observatory.yaml` (`TensorOperator`),
  the committed observatory JSON Schema, `crates/modelspec-observatory/`, `ui/src/generated/`.
- `DESIGN.md` § 2.2 (the `i32` rule), § 4 (the gradient rule), § 8.1 (`E-DATA-004`), the revision
  line; `website/docs/reference/` through `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`embedding` over an `i32` id input checks, lowers, runs on the CPU interpreter and passes an `f64`
finite-difference gradient check, and every other use of `i32` is refused with a code.

Scenarios, each run by `task ci` through the command shown:

1. `cd spec && ess specify validate --path .` exits 0 with `I32` in `modelspec.checkpoint.DType`
   and `Embedding` and `EmbeddingGrad` in `modelspec.observatory.TensorOperator`.
2. `cargo test -p modelspec-runtime --test gradients_embedding`: for `V = 7`, `H = 3`, `N = 10`
   and ids that repeat one row three times and leave two rows unread, the gradient of an `mse`
   loss over `embedding(table, ids)` with respect to `table` is within `TOL_F64` of central
   differences in `f64` and within `TOL_F32` in `f32`, and the gradient of the two unread rows is
   exactly 0. `cargo test -p modelspec-runtime --test gradients` passes on the moved harness.
3. `cargo test -p modelspec-runtime --test gradients_embedding positional`: a loss over
   `embedding(tok, ids) + embedding(pos, positions)` passes the same check for both tables.
4. `cargo test -p modelspec-cli --test check_i32`: `modelspec check` on fixtures that add an `i32`
   to an `f32`, declare an `i32` `param`, and return an `i32` from a function exits 1 each, with
   the new `E-TYPE` code at the offending span; a fixture whose `ids: Tensor<i32, [4]>` is read only
   by `embedding` exits 0.
5. `cargo test -p modelspec-runtime embedding_id_out_of_range`: ids containing `V` and `-1` give
   the out-of-range `RuntimeError` of the interpreter, not a panic.
6. `cargo test -p modelspec-cli --test train_i32_synthetic`: `modelspec train --backend cpu` on a
   fixture with an `i32` input exits 1 with `E-DATA-004`.
7. `task docs-check` exits 0, and `website/docs/reference/operators.md` lists `embedding` and
   `website/docs/reference/diagnostics.md` lists `E-DATA-004`.

## Out of scope

- Other integer types, integer arithmetic, casts.
- A token-type (segment) embedding: the BERT-tiny configuration of `story:bert-tiny-cpu-model` has
  none, because the epic trains MLM only.
- The Vulkan kernel (`story:bert-vulkan-kernels-rowwise`).
