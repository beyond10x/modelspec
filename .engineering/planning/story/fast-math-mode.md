---
format: aep.planning-md/3
id: story:fast-math-mode
kind: story
status: draft
title: 'An opt-in fast-math mode: contraction, cooperative matrices and a parallel loss sum'
relations:
- depends_on: story:matmul-shape-aware-tiling
- depends_on: story:vulkan-loss-slots-written-by-kernel
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/compile.rs
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: inferred
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
revision: 7
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
`fast_math` off. The setting is the existing `fast_math` field (`KernelCacheKey`,
`crates/modelspec-core/src/artifact.rs:247`; `LoadedArtifact`, `artifact.rs:271`). Nothing sets it
today: `compile` copies it from the cache key (`crates/modelspec-cli/src/compile.rs:184`), and every
constructor writes `false` (`artifact.rs:1329`, `crates/modelspec-runtime/src/spirv/emit.rs:245`).
This story sets it with a `--fast-math` flag on `compile` and `train`.

The multi-workgroup loss sum rewrites the loss kernel that story:vulkan-loss-slots-written-by-kernel
makes write the step's loss slot, so this story follows that one.

**ADR gate.** DESIGN.md § 12 ADR-008 (fast-math mode) states what `fast_math` permits (contraction,
cooperative matrices, a parallel reduction order), what it is compared against and within which
tolerance, and that benchmarks keep reporting the deterministic figures.

Outcome: a program compiled with `--fast-math` emits its matmul's float multiplies and adds without
`NoContraction`, may use cooperative matrices where the device profile reports them, and sums the
loss over more than one workgroup; without the flag nothing changes.

## Acceptance

- **adr-gate**: DESIGN.md § 12 ADR-008 (fast-math mode) is accepted before the story moves to
  proposed; a draft of it is on its own PR.
- **flag-sets-field**: `compile --fast-math` and `train --fast-math` set `fast_math`; a test
  compiles one file with and without the flag, reads `fast_math` back through the artifact reader
  as `true` and `false`, and finds the two artifact hashes different.
- **emission-checked**: a test reads the emitted SPIR-V and finds no `NoContraction` on the matmul
  kernel's `OpFMul` and `OpFAdd` under `fast_math`, and finds it on all of them without it.
- **cooperative-matrix-emitted**: compiled with `--fast-math` for a `--device-profile` that reports
  cooperative-matrix support, the matmul module declares the `CooperativeMatrixKHR` capability; for
  a profile without that support it does not.
- **parallel-loss-sum**: under `fast_math` the loss-sum kernel of the MNIST plan has more than one
  workgroup in `ir --stage kernel`; without it, it keeps `[1, 1, 1]`.
- **default-unchanged**: with `fast_math` off, the artifact hash and every bit of the MNIST run are
  unchanged.
- **fast-math-tolerance**: the loss curve under `fast_math` stays within ADR-008's tolerance of the
  CPU oracle.
- **numerics-class-reported**: `train --backend vulkan --stats --fast-math` prints `numerics class: tolerance`;
  `numerics_class` (`crates/modelspec-runtime/src/vulkan/runner.rs:1073`) reads the artifact's `fast_math`
  field, which today it does not.
- **fast-math-throughput** (reported, not gated): `mlp_sweep` at b8192 h4096 and per-matmul GFLOP/s
  from `--profile`, with and without `fast_math`, are recorded in the story's closing evidence.
- `task ci` exits 0.

## Scope

- `DESIGN.md`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-core/src/artifact.rs`
- `crates/modelspec-cli/src/compile.rs`
- `crates/modelspec-cli/src/train.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`

## Out of scope

- Any deterministic-mode change; deterministic tiling is story:matmul-shape-aware-tiling, which this
  story builds on.
- Writing the loss into its step slot, which story:vulkan-loss-slots-written-by-kernel holds and
  this story follows.
- Benchmark figures with `fast_math` on as the headline number.
