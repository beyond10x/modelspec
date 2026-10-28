---
format: aep.planning-md/3
id: story:decider-small-operators
kind: story
status: proposed
title: 'Small operators: reshape, concat, smooth_l1, sqrt, abs, masked cross_entropy, negative literals'
summary: Seven source-language additions, each gradient-checked and with a Vulkan kernel held to the CPU reference.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/check_small_ops.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ops.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-observatory/schema
- confidence: cited
  path: crates/modelspec-runtime/schema
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/gradients_small_ops.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_small_ops.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T12:12:30Z", actor: "human:timo", revision: 5}
---
# Small operators: reshape, concat, smooth_l1, sqrt, abs, masked cross_entropy, negative literals

`depends_on: none`

## Outcome

Seven additions to the source language, each on the CPU interpreter and with a Vulkan kernel held
to the CPU reference, each differentiable where it takes an `f32` operand:

1. `reshape(x, shape = [d1, …])`: row-major, the element count unchanged; its gradient is the
   reverse reshape. A count mismatch is a new E-TYPE code at the call.
2. `concat(a, b)`: on the last axis of two rank-2 operands with equal leading dimension; its
   gradient is the split. Other ranks or a leading-dimension mismatch is refused at the call.
3. `smooth_l1(a, b, beta = F)`: the mean over all elements of the Huber-style smooth-L1 of
   `a − b` with threshold `beta > 0`, a rank-0 result like `mse`.
4. `sqrt(x)`: elementwise, gradient `0.5 / sqrt(x)`.
5. `abs(x)`: elementwise, gradient `sign(x)`, `0` at `0`.
6. `cross_entropy(logits, labels, weights, mask)`: an optional fourth operand, `f32` of the
   logits' shape, `1` for a class that takes part and `0` for one that does not; a masked class
   gets probability 0 and gradient 0. A weighted row whose label is a masked class gives a
   non-finite loss, which the existing non-finite refusal reports.
7. A negative float literal as an operand (`mul(d, -1.0)`), not only as a named argument; a unary
   minus on anything but a float literal stays E-SYNTAX-009.

## Context

- Gap rows 5–10, and the workarounds `decider.mspec` carries for them: two selection matmuls and a
  broadcast matmul for the reshape (`fold_rows`, `fold_cols`, `fold_bcast`; ≈ 138 MFLOP a step and
  4.3 MB of data for `fold_rows` alone), a split first-layer weight for the concatenation,
  smooth-L1 from `relu`, `sub` and `mul`, an additive `-1e9` bias for absent candidates,
  `sub(0.0, x)` for a sign.
- Decider design § 5.2 (predictor input `s_t` concatenated with a candidate action embedding; the
  head over `[s_t ; ŝ ; ‖…‖]`), § 5.3 step 1 (smooth-L1 in embedding space), step 2
  (cross-entropy over candidates; not every template applies to every decision, § 4.3).
- Source operators: `OPERATORS` in `crates/modelspec-core/src/resolve.rs`; shape and gradient
  rules in `crates/modelspec-core/src/ops.rs`; `TensorOp` in `ir/tensor.rs` already has an
  internal `Sqrt` (the AdamW subgraph, `ir/tensor.rs:240`). A negative literal is parsed only after
  `name =` today (`crates/modelspec-core/src/syntax/parser.rs:894-899`).
- The planned `story:jepa-stop-gradient` touches the same files (`resolve.rs`, `ops.rs`,
  `ir/tensor.rs`, `spec/domains/observatory.yaml`); it is not in this store's main yet, so
  `waves` cannot see the collision. Land them one after the other.

## Spec first

`spec/domains/observatory.yaml` `TensorOperator` gains the variants the implementation adds
(the forward operators and their internal gradient operators) before any code; then `task
spec-validate`, `task schema-generate` (both `schema/` trees) and `task ui-types`
(`ui/src/generated/`).

## Scope

- `crates/modelspec-core/src/`: `resolve.rs`, `ops.rs`, `types.rs`, `ir/tensor.rs`,
  `ir/print.rs`, `ir/kernel.rs`, `schedule.rs`, `artifact.rs`, `autodiff.rs`,
  `syntax/parser.rs`.
- `crates/modelspec-runtime/src/cpu/interpreter.rs`, `crates/modelspec-runtime/src/spirv/emit.rs`,
  `crates/modelspec-runtime/src/spirv/emit/rowwise.rs`.
- `spec/domains/observatory.yaml`, `crates/modelspec-observatory/schema/`,
  `crates/modelspec-runtime/schema/`, `ui/src/generated/`.
- Tests (new): `crates/modelspec-runtime/tests/gradients_small_ops.rs`,
  `crates/modelspec-runtime/tests/vulkan_small_ops.rs`,
  `crates/modelspec-cli/tests/check_small_ops.rs` with fixtures under
  `crates/modelspec-cli/tests/fixtures/`.
- `DESIGN.md` § 2.2, § 4, § 8.1; `crates/modelspec-docs/src/operators.rs`, then `task
  docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

- **gradients-small-ops**: `cargo test -p modelspec-runtime --test gradients_small_ops` checks the
  gradient of every `f32` operand of `reshape`, `concat`, `smooth_l1` (on both sides of `beta`),
  `sqrt` (on positive inputs), `abs` (away from 0) and masked `cross_entropy` against central
  differences, within `TOL_F64` in `f64` and `TOL_F32` in `f32`; the masked classes' gradient is
  exactly 0.
- **vulkan-small-ops**: `cargo test -p modelspec-runtime --features vulkan --test vulkan_small_ops
  -- --ignored` runs each forward and gradient kernel on the device, each within the DESIGN.md
  § 9 tolerance of the CPU reference; `spirv_val` (in `task test`) validates every new module.
- **check-small-ops**: `cargo test -p modelspec-cli --test check_small_ops`: a reshape whose
  element count differs, a `concat` of rank-1 operands or of unequal leading dimensions, `beta =
  0.0`, and a mask whose shape is not the logits' are each refused at the call with their code;
  `mul(d, -1.0)` checks and evaluates to `-d`; `mul(-x, y)` with `x` a name stays E-SYNTAX-009.
- **small-ops-equal-workarounds**: in the same test, a fixture computing each of reshape, concat,
  smooth-L1 and masked cross-entropy both ways (the operator, and the 0.21.0 workaround the decider
  file uses) gives the same loss on the CPU, bit-identical where the arithmetic is identical and
  within `TOL_F32` otherwise.
- `task ci` exits 0, including `task docs-check`, `task spec-validate`, `task schema-check` and
  `task ui-types-check`.

## Out of scope

- `stop_gradient`, `sin`, `cos`, SIGReg: `epic:jepa-operators`.
- `concat` on other axes or of more than two operands; `reshape` to a symbolic shape.
- `min`, `max`, `where`, a general unary minus on expressions.
- Rewriting the decider fixture: the epic's closing check.
