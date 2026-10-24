---
format: aep.planning-md/3
id: story:mean-mse-adjoints-divide-by-n
kind: story
status: draft
title: Mean and mse adjoint scale documented as a folded reciprocal
scope:
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-core/tests/adjoint_scale.rs
revision: 8
---
# Mean and mse adjoint scale documented as a folded reciprocal

## Why

Review of 0.5.0 (finding C4), still open at main 496e98e. The adjoints of `mean` and `mse` multiply
by a scale folded at compile time, `Scalar(1/n)` and `Scalar(2/n)`
(`crates/modelspec-core/src/ops.rs:261-285`), computed in `f64` and rounded once to `f32`
(`ops.rs:325-335`). DESIGN.md § 4 states them as divisions, `G / numel(A)` and
`2 * (Y-T) / numel(Y)` (DESIGN.md:245-246). For an `n` that is not a power of two, the product with
the folded reciprocal can differ from a true division by up to 1 ULP.

The code keeps the folded reciprocal (operator's decision). DESIGN.md § 5.4 puts "`mul` by the
gradient scale of `mse` (folded to a constant at compile time)" in the bit-for-bit class
(DESIGN.md:303). A division op would put every gradient that reads it in the tolerance class
(`OpFDiv`, DESIGN.md:304). So the specification changes to state what the code does.

Outcome: DESIGN.md § 4 states the `mean` and `mse` adjoint scale as a folded reciprocal, and a test
holds the code to it for both.

## Acceptance

- **design-states-folded-reciprocal**: the § 4 rows for `mean` and `mse` (DESIGN.md:245-246) state
  the adjoint as `G * c`, `c = f32(1 / numel(A))`, and `dY = (Y-T) * (G * c)`,
  `c = f32(2 / numel(Y))`, each `c` folded at compile time. The text names the difference of at
  most 1 ULP from a true division when `n` is not a power of two, and says that the CPU oracle and
  the GPU multiply by the same constant.
- **mean-adjoint-bits-n3**: a CPU test differentiates `mean` over 3 elements and asserts that the
  adjoint's bits equal `G * (1.0f32 / 3.0)`. The core crate has no evaluator, so the test reads the
  folded `Scalar` (its bits equal `(1.0f32 / 3.0).to_bits()`) and the `Mul` of `G` by it.
- **mse-adjoint-bits-n3**: a CPU test differentiates `mse` over 3 elements and asserts that the
  folded `Scalar`'s bits equal `f32(2/3)`, and that `dY` is `Mul(d, Mul(G, c))` with `d = Sub(Y, T)`,
  the order at `ops.rs:270-277`.
- `task ci` exits 0.

## Scope

- `DESIGN.md`
- `crates/modelspec-core/tests/adjoint_scale.rs` (new)

## Out of scope

- A division op in the tensor IR, and any change to `ops.rs`: the gradient bits stay as they are.
- The `mean` and `mse` values, which already pass through `OpFDiv` in the tolerance class
  (DESIGN.md:304).
