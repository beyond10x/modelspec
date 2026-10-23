---
format: aep.planning-md/3
id: story:fast-math-mode
kind: story
status: draft
title: 'An opt-in fast-math mode: contraction, cooperative matrices and a parallel loss sum'
relations:
- depends_on: story:matmul-shape-aware-tiling
scope:
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
revision: 3
---
# An opt-in fast-math mode: contraction, cooperative matrices and a parallel loss sum

## Why

Review of 0.7.1 (finding 3, its FMA and cooperative-matrix part, and finding 5), at main 496e98e.
Every float multiply and add carries `NoContraction`
(`crates/modelspec-runtime/src/spirv/emit.rs:917-919`), so no FMA is formed, and the loss sum runs in
one workgroup (`[1, 1, 1]`, `crates/modelspec-core/src/schedule.rs:456`). Both hold the deterministic
bits; both cost throughput.

DESIGN.md:228 already names `fast_math = true` as a separate opt-in compilation setting recorded in
the artifact hash, and the benchmark rule (DESIGN.md:616) runs ModelSpec with `deterministic` on and
`fast_math` off.

**This story needs an ADR in DESIGN.md § 12 before it can be proposed** (operator's decision). The
ADR states what `fast_math` permits (contraction, cooperative matrices, a parallel reduction order),
what it is compared against and within which tolerance, and that benchmarks keep reporting the
deterministic figures.

Outcome: a program compiled with `fast_math = true` may use FMA, cooperative matrices and a
multi-workgroup loss sum; without it nothing changes.

## Acceptance

- **adr-recorded**: DESIGN.md § 12 holds the accepted ADR before the story moves.
- **default-unchanged**: with `fast_math` off, the artifact hash and every bit of the MNIST run are
  unchanged.
- **fast-math-hash**: `fast_math = true` changes the artifact hash.
- **fast-math-throughput**: with `fast_math = true`, `mlp_sweep` at b8192 h4096 and per-matmul
  GFLOP/s from `--profile` are reported against the deterministic figures.
- **fast-math-tolerance**: the loss curve under `fast_math` stays within the ADR's tolerance of the
  CPU oracle.
- `task ci` exits 0.

## Scope

- `DESIGN.md`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/kernel.rs`

## Out of scope

- Any deterministic-mode change; deterministic tiling is story:matmul-shape-aware-tiling, which this
  story builds on.
- Benchmark figures with `fast_math` on as the headline number.
