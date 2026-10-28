---
format: aep.planning-md/3
id: story:jepa-sigreg-cpu
kind: story
status: proposed
title: sigreg on the CPU interpreter, gradient-checked
summary: A built-in SIGReg loss operator in characteristic-function form with per-step projections, finite-difference gradient check and a run that lowers the statistic.
relations:
- decomposes: epic:jepa-operators
- depends_on: story:jepa-stop-gradient
- serves: vision:modelspec
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli
- confidence: cited
  path: crates/modelspec-core/src
- confidence: cited
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu
- confidence: cited
  path: crates/modelspec-runtime/src/train
- confidence: cited
  path: crates/modelspec-runtime/tests
- confidence: cited
  path: schema
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 13
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T10:45:52Z", actor: "human:timo", revision: 2}
---

`id: story:jepa-sigreg-cpu` · depends_on: `story:jepa-stop-gradient` (shared files: `resolve.rs`,
`ops.rs`, `ir/tensor.rs`, `types.rs`, `observatory.yaml`, the generated schemas, DESIGN.md) ·
size: about one day

## Outcome

`sigreg(z, slices = M)`, with `z: f32[N, K]`, returns `f32[]`: the Epps–Pulley statistic of each
of `M` random unit projections of the rows of `z`, averaged over the projections. It checks, lowers,
runs and trains on the CPU interpreter, and its gradient passes a finite-difference check in `f64`.
The projections are redrawn every step on a new Philox stream, as dropout's mask is.

Semantics, recorded in DESIGN.md § 2.2, § 2.3 and § 4:

- **Lowering.** A call becomes `ReduceMean(SigregColumns(MatMul(z, proj)))`. `proj: f32[K, M]` is
  an internal `Input` named `<Model>.<fn>.sigreg#<k>`, with the dropout site naming and inlining
  rule (DESIGN.md:142).
- **Projections.** Stream tag 6. Draw id: `blake3(site bytes ++ step as u64 LE)`, as `dropout_id`
  does. Element `i = k·M + m` is the existing Box-Muller standard-normal draw. Each column is
  divided by its L2 norm in `f64` and rounded to `f32` once. The training host supplies the matrix
  each step. A held-out evaluation and `infer` use the matrix of the step being evaluated, or of the
  checkpoint's step.
- **`SigregColumns(P: [N, M]) -> [M]`.** Knots `t_j = 0.625·j` for `j = 1..8`. These are the
  positive half of 17 knots on [-5, 5]; the knot at 0 contributes exactly 0. `φ_j = exp(-t_j²/2)`
  and the weights `c_j = 1.25·φ_j` (`j < 8`) and `c_8 = 0.625·φ_8` are `f32` constants computed in
  `f64`. For column `m`: `C_j = Σ_n cos(t_j P[n,m]) / N` and `S_j = Σ_n sin(t_j P[n,m]) / N`, each
  sum in ascending `n` from its first term. The column value is
  `T_m = N · Σ_j c_j · ((C_j − φ_j)² + S_j²)`.
- **`SigregColumnsGrad(P, G: [M]) -> [N, M]`.** It recomputes `C_j` and `S_j` and gives
  `dP[n,m] = 2·G[m]·Σ_j c_j t_j (S_j cos(t_j P[n,m]) − (C_j − φ_j) sin(t_j P[n,m]))`. `MatMul`'s
  existing rule gives `dz = dP · transpose(proj)`. `proj` takes no gradient.
- **`sin` and `cos`.** Both are evaluated by one written-out `f32` routine: Cody–Waite reduction by
  π/2 with a three-part constant, then fixed minimax polynomials, each operation rounded and each
  product after the first fused with `mul_add`. It is not the platform `sinf`, so S3 can emit the
  identical sequence. Vulkan's `GLSL.std.450 Sin` only promises 2^-11 absolute error on [-π, π],
  far outside `GPU_ATOL = 1e-5` (DESIGN.md:598). This follows the GELU precedent of one shared
  evaluation (DESIGN.md:282).

## Context

- The dropout site, token and step draw are the template: `lower.rs:326` (`token`), `:355` (site
  name), `ir/training.rs:550` (`dropout_sites`), `train/init.rs:39` (`Stream`), `:129`
  (`dropout_id`), `:140` (`dropout_token`), `:181` (`draw`), `:219` (`standard_normal`),
  `train/runner.rs:409` and `:430` (per-step supply).
- Named compile-time argument: `ops::named_parameters` / `with_named` (`ops.rs:162-180`).
  `NamedKind::PositiveInteger` already exists (`ops.rs:84`).
- A per-column internal op on the CPU: `SumRows` (`ops.rs:402`, `cpu/interpreter.rs:896`).
- The finite-difference harness: `crates/modelspec-runtime/tests/common/finite_difference.rs` and
  `common/mod.rs` (`TOL_F64`, `TOL_F32`); `gradients_dropout.rs` is the closest precedent.

## Scope

- `crates/modelspec-core/src/resolve.rs`, `ops.rs` (`Sigreg { slices }` source op,
  `SigregColumns`, `SigregColumnsGrad`), `ir/tensor.rs`, `ir/print.rs`, `ir/verify.rs`,
  `ir/training.rs` (`sigreg_sites`), `lower.rs`, `types.rs` (`z` rank 2: E-TYPE-004; `i32` `z`:
  E-TYPE-018).
- `crates/modelspec-runtime/src/cpu/interpreter.rs`, `cpu/tensor.rs` (the shared `sin_cos`
  routine), `train/init.rs` (`Stream::Projection` = 6, `sigreg_projection`), `train/runner.rs`,
  `train/eval.rs`.
- `crates/modelspec-cli/src/infer.rs` (projections of the checkpoint's step).
- Tests: `crates/modelspec-runtime/tests/gradients_sigreg.rs` (new),
  `crates/modelspec-runtime/tests/sin_cos.rs` (new), `crates/modelspec-cli/tests/check_sigreg.rs` and
  `train_sigreg.rs` (new) with fixtures under `crates/modelspec-cli/tests/fixtures/`.
- `spec/domains/observatory.yaml:203-204` (`Sigreg`, `SigregColumns`, `SigregColumnsGrad`), then
  `task schema-generate` and `task ui-types`.
- DESIGN.md § 2.2, § 2.3 (stream tag 6), § 4 (rule rows; the `sin`/`cos` routine with its
  constants) and the revision line. `crates/modelspec-docs/src/operators.rs`, then
  `task docs-generate`. One line in `CHANGELOG.md`.

## Acceptance

`cargo test -p modelspec-runtime --test gradients_sigreg` passes:

- The gradient of `sigreg(matmul(x, w), slices = 16)` with respect to `w` (`N = 33`, `K = 8`)
  matches central differences in `f64` within `TOL_F64`, and in `f32` within `TOL_F32`. The
  projections are fixed to step 0 of a named site.
- Its value equals an independent `f64` evaluation of the formula above, written in the test, within
  `1e-5` relative.

## Done when

- `cargo test -p modelspec-runtime --test sin_cos`: the routine is within 2 `f32` ulp of `f64`
  `sin` and `cos` at every multiple of 1/256 in [-64, 64], and returns the same bits on every call.
- `cargo test -p modelspec-cli --test train_sigreg`: `modelspec train --backend cpu` runs a fixture
  whose loss is `sigreg` of a `[256, 8]` `xavier_uniform` parameter alone, under `sgd`. Two runs
  from one seed print identical lines, and the last loss is below a tenth of the step-0 loss. That
  threshold is unmeasured: if the ratio is not reached, record it as a finding.
- `check_sigreg`: `z` of rank 1, an `i32` `z`, and `slices = 0` each exit 1 with their code at the
  call.
- `task ci` is green.

## Out of scope

- Vulkan kernels (S3). Until S3, `--backend vulkan` on a `sigreg` program answers E-TARGET-011 at
  the call, as `muon` does.
- `sin`, `cos`, `exp` or `abs` as source operators, and smooth-L1. The decider design's loss can
  use `mse`.
- Fixed (once-per-run) projections, and `knots` as an argument.

---

