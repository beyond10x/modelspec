---
format: aep.planning-md/3
id: story:fast-math-mode
kind: story
status: draft
title: 'An opt-in fast-math mode: contraction under --fast-math'
relations:
- depends_on: story:matmul-shape-aware-tiling
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/compile.rs
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-runtime/benches/mlp_sweep.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: website/docs/reference/cli.md
revision: 12
---
# An opt-in fast-math mode: contraction under `--fast-math`

## Why

Review of 0.7.1 (finding 3, its FMA part), at main 496e98e. Every float multiply and add carries
`NoContraction` (`crates/modelspec-runtime/src/spirv/emit.rs:917-919`), so no FMA is formed. That
holds the deterministic bits and costs throughput: the register-blocked matmul of
story:matmul-shape-aware-tiling reaches 9,588 GFLOP/s on the forward site at batch 8192, hidden
4096 (`benchmarks/raw/2026-10-24-tiling/after-profile-b8192-h4096.txt`) with a separate multiply
and add per product, where a fused multiply-add is one instruction.

DESIGN.md:228 already names `fast_math = true` as a separate opt-in compilation setting recorded in
the artifact hash, and the benchmark rule (DESIGN.md:616) runs ModelSpec with `deterministic` on and
`fast_math` off. The setting is the existing `fast_math` field (`KernelCacheKey`,
`crates/modelspec-core/src/artifact.rs:247`; `LoadedArtifact`, `artifact.rs:271`). Nothing sets it
today: `compile` copies it from the cache key (`crates/modelspec-cli/src/compile.rs:184`), and every
constructor writes `false` (`artifact.rs:1329`, `crates/modelspec-runtime/src/spirv/emit.rs:245`).
This story sets it with a `--fast-math` flag on `compile` and `train`. The CLI reference page
`website/docs/reference/cli.md` is generated from the clap definitions
(`crates/modelspec-docs/src/main.rs:5`, `:212`) and drift-checked by `task check`, so it is
regenerated with the flag.

`numerics_class` (`crates/modelspec-runtime/src/vulkan/runner.rs:1068-1080`) decides the class from
the negotiation and the plan's kernel kinds only. Every M1 training run already prints
`numerics class: tolerance`, so the printed line cannot show the change; the function itself is
tested.

**Split on 2026-10-24 (wave coordinator, decision D12 on `.engineering/waves/perf-1.md`).** The
story as first drafted also covered cooperative-matrix matmuls and a multi-workgroup loss sum.
The RTX 3090 reports eleven cooperative-matrix shapes through `VK_KHR_cooperative_matrix`
(`inspect-device --backend vulkan --format json`, 2026-10-24) and none takes `f32` operands: the
`a_type`/`b_type` pairs are `FLOAT16`, `BFLOAT16`, `SINT8` and `UINT8`, with `FLOAT32` only as
`c_type`/`result_type`. ADR-008 permits no reduced-precision operand or accumulation, so a
cooperative-matrix matmul cannot run on this device under ADR-008 as accepted; that part is
story:fast-math-cooperative-matrix, blocked by decision-blocker:cooperative-matrix-f32-operands.
The multi-workgroup loss sum rewrites the loss kernel that
story:vulkan-loss-slots-written-by-kernel changes, so it is story:fast-math-parallel-loss-sum
after that one. What stays here is the flag, the artifact field, contraction, the numerics class
and the measurement.

**ADR gate.** DESIGN.md § 12 ADR-008 (fast-math mode) states what `fast_math` permits (contraction,
cooperative matrices, a parallel reduction order), what it is compared against and within which
tolerance, and that benchmarks keep reporting the deterministic figures. Accepted 2026-10-24
(DESIGN.md revision 0.29).

Outcome: a program compiled with `--fast-math` emits the float multiplies, adds and subtracts of
its matmul, reduction, loss-sum and SGD kernels without `NoContraction`, so the driver may form
fused multiply-adds; without the flag nothing changes.

## Acceptance

- **adr-gate**: DESIGN.md § 12 ADR-008 (fast-math mode) is accepted before the story moves to
  proposed (satisfied: revision 0.29, 2026-10-24).
- **flag-sets-field**: `compile --fast-math` and `train --fast-math` set `fast_math`; a test
  compiles one file with and without the flag, reads `fast_math` back through the artifact reader
  as `true` and `false`, and finds the two artifact hashes different.
- **cli-reference-regenerated**: `website/docs/reference/cli.md` lists `--fast-math` on `compile`
  and `train`, and its drift check in `task check` passes.
- **emission-checked**: a test reads the emitted SPIR-V and finds no `NoContraction` on any
  `OpFMul`, `OpFAdd` or `OpFSub` of the matmul, reduction, loss-sum and SGD modules under
  `fast_math`, and finds it on all of them without it.
- **default-unchanged**: with `fast_math` off, the artifact hash and every bit of the MNIST run are
  unchanged (the goldens and the same-bits device tests pass unchanged).
- **fast-math-tolerance**: `train --backend vulkan --fast-math` on `examples/mnist_mlp.mspec` and on
  the sweep source at b8192 h4096 passes against the CPU oracle within ADR-008's tolerance
  (`GPU_ATOL = 1e-5`, `GPU_RTOL = 1e-4`, whole-run L2 bound `1e-4`), and two runs of one
  `fast_math` artifact on this device give the same bits.
- **numerics-class-follows-fast-math**: a unit test calls `numerics_class` with a bit-for-bit
  negotiation and a plan whose kernels are all matmuls; it returns `BitForBit` when the artifact's
  `fast_math` is off and `Tolerance` when it is on, and the run record of a `fast_math` run names
  the class `tolerance`.
- **fast-math-throughput** (reported, not gated): `mlp_sweep` at b8192 h4096, b8192 h1024 and b1000
  h4096 and per-matmul GFLOP/s from `--profile` at b8192 h4096, with and without `fast_math`, are
  recorded in the story's closing evidence beside the PyTorch eager CUDA medians of
  `benchmarks/runs.yaml`; the deterministic figure stays the headline. `mlp_sweep` takes a
  `--fast-math` switch so the figure is reproducible.
- `task ci` exits 0.

## Scope

- `DESIGN.md`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-core/src/artifact.rs`
- `crates/modelspec-cli/src/compile.rs`
- `crates/modelspec-cli/src/train.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-runtime/benches/mlp_sweep.rs`
- `website/docs/reference/cli.md` (generated)

## Out of scope

- Any deterministic-mode change; deterministic tiling is story:matmul-shape-aware-tiling, which this
  story builds on.
- Cooperative-matrix matmuls: story:fast-math-cooperative-matrix.
- The multi-workgroup loss sum: story:fast-math-parallel-loss-sum, after
  story:vulkan-loss-slots-written-by-kernel.
- Benchmark figures with `fast_math` on as the headline number.
