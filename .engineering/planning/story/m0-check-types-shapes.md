---
format: aep.planning-md/3
id: story:m0-check-types-shapes
kind: story
status: active
title: Name resolution, type and shape checking, and modelspec check
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
- depends_on: story:m0-parse-subset
- depends_on: story:m0-tensor-ir-core
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: Taskfile.yml
- confidence: cited
  path: crates/modelspec-cli/src/check.rs
- confidence: cited
  path: crates/modelspec-cli/tests/check.rs
- confidence: cited
  path: crates/modelspec-cli/tests/check_adversary.rs
- confidence: cited
  path: crates/modelspec-cli/tests/exit_codes.rs
- confidence: cited
  path: crates/modelspec-core/src/hir.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: tests/language/check
revision: 13
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:26Z", actor: "human:timo", revision: 10, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 11, decided_on: {"recorded":{"review_outcome":2}}}
---
# Name resolution, type and shape checking, and `modelspec check`

## Context

Turns the Source AST into Typed HIR and reports user specification errors before anything runs
(DESIGN.md § 0 invariant 3: "The compiler rejects invalid operations before dispatch"). Implements
DESIGN.md § 3 (name resolution, tensor type and shape checks, Typed HIR; the effect check is
owned by story `m0-cpu-sgd-training`, because M0 source has no effects before Training IR), § 3.1 (Typed HIR:
resolved types, dimension bindings, spans, stable parameter paths), § 3.3 passes 1 and 2, § 2.2
shape rules (through the operator registry), § 2.3 canonical parameter paths, § 8 verb `check` and
§ 8.1 diagnostics. Owns the first epic acceptance bullet.

## Lands in

- `crates/modelspec-core/src/resolve.rs`: scopes for `param`, function arguments, `let`, the `train`
  block names; `model.forward` resolved to the model function; unknown and duplicate names are
  `E-NAME-0nn` diagnostics.
- `crates/modelspec-core/src/types.rs`: types every expression through `ops.rs` (read only; this
  story does not edit the registry) and maps registry errors to `E-TYPE-0nn` diagnostics with spans.
  `E-TYPE-012` is the `matmul` contracting mismatch, rendered in the exact layout of DESIGN.md § 8.1
  (code and message, `path:line:col`, `lhs:`, `rhs:`, `expected rhs dimension[0] = K; got N`).
  Train-block rules: `model =` names the model; `forward` argument and return types match its
  declaration; `loss` is rank 0; `optimizer` is `sgd` with a positive finite `lr`; `steps` positive;
  `seed` fits `u64`; `param` initializer is `xavier_uniform()` on a rank-2 parameter; zero
  dimensions rejected.
- `crates/modelspec-core/src/hir.rs`: Typed HIR, each parameter carrying its canonical path
  (`LinearReg.w`, DESIGN.md § 2.3), and the checked `train` block as `HirTrain`: the model, the
  `input` and `target` declarations with their types, the `loss` expression, `lr`, `steps`, `seed`
  and `device`. Later stories read the train configuration from `HirTrain` and do not edit this
  file.
- `crates/modelspec-cli/src/check.rs`: `modelspec check <file>` exits 0 with no output on success,
  1 after printing every diagnostic to stderr.
- `tests/language/check/`: one fixture per new code, among them `matmul_mismatch.mspec` reproducing
  the § 8.1 example (lhs `[64, 32]`, rhs `[64, 128]`).
- `Taskfile.yml`: task `check-examples` running `modelspec check examples/linear_regression.mspec`,
  added to `ci`; AGENTS.md notes that `examples/tiny_mlp.mspec` joins it when modules and generics
  land (AGENTS.md rule on examples).

## Out of scope

- Lowering to Tensor IR and `modelspec ir` (story `m0-lower-tensor-ir`).
- Modules, layers, generics and `examples/tiny_mlp.mspec` (DESIGN.md § 11 iteration 3).
- Device capability checks: `device = vulkan(N)` is accepted syntactically and not validated.
- Initializers other than `xavier_uniform()`.

## Domain relations

None. Types, shapes and parameter paths of the `.mspec` language are not an ESS domain (AGENTS.md).

## Acceptance

`modelspec check examples/linear_regression.mspec` exits 0, and
`modelspec check tests/language/check/matmul_mismatch.mspec` exits 1 printing
`E-TYPE-012: matmul contracting dimension mismatch` followed by the location, `lhs`, `rhs` and
`expected rhs dimension[0] = 32; got 64` lines of DESIGN.md § 8.1, both asserted by
`cargo test -p modelspec-cli --test check`.
