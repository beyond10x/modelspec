---
format: aep.planning-md/3
id: story:deterministic-fma
kind: story
status: active
title: Deterministic kernels use an explicit fused multiply-add (ADR-009)
relations:
- serves: vision:modelspec
- depends_on: story:matmul-tile-from-device-invocation-limit
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T18:58:59Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-24T18:58:59Z", actor: "human:timo", revision: 3}
---
# Deterministic kernels use an explicit fused multiply-add (ADR-009)

## Context

ADR-009 (DESIGN.md § 12, accepted 2026-10-24 under Timo's grant of the same day): in
`deterministic` mode every product-and-sum of a matmul, reduction or loss-sum kernel is one IEEE
754 fused multiply-add, `OpExtInst GLSL.std.450 Fma`, and the CPU oracle computes the same
element with `f32::mul_add`. Today a deterministic kernel emits `OpFMul` then `OpFAdd`, both
decorated `NoContraction` (`crates/modelspec-runtime/src/spirv/emit.rs:10`), and the CPU
backend multiplies and adds separately (no `mul_add` in `crates/`). Two instructions per
product-and-sum halve the FP32 ceiling of the GA102 against its fused peak; the forward matmul
`[8192, 784] x [784, 4096]` of `mlp_sweep` b8192 h4096 runs at 9.2 TFLOP/s deterministic and
11.4 under `fast_math` (`benchmarks/raw/2026-10-24-tile256/padonly-*`, medians of 3 profiled
runs of 220 steps), and the step at 12,313.7 µs against 7,290.5 for PyTorch eager CUDA.

## Outcome

Every deterministic kernel that accumulates products emits `Fma` for each product-and-sum, the
CPU oracle uses `f32::mul_add` at the same places, CPU and Vulkan agree bit for bit on every
gate, and every recorded bit (goldens, plan-site digests, `train` stdout hashes, checkpoint
fixtures) is re-recorded once in this unit with the before and after values kept side by side.
The `k = 0` product keeps its role: the first fma takes the accumulator's seed value, which is
the plan's documented start (the `k = 0` product computed as `fma(a0, b0, 0.0)` or the plain
product; the unit states which and the oracle matches it). `fast_math` keeps ADR-008's meaning.

## Acceptance

- **fma-emitted**: for every kernel kind that accumulates products (matmul in every tile
  including the register-blocked tiles, the transposed-operand staging paths and the
  256-invocation block; reductions; the `mse` loss sum), the SPIR-V arithmetic section holds one
  `OpExtInst Fma` per product-and-sum and no `OpFMul` followed by an `OpFAdd` of its result;
  every remaining `OpFAdd`, `OpFSub` and `OpFMul` keeps `NoContraction`; `spirv-val` passes; a
  test reads the modules of the MNIST step plan and the sweep plans and counts them.
- **oracle-matches**: the CPU backend computes the same elements with `f32::mul_add` in the same
  order; `cargo test` CPU-against-Vulkan bit comparisons (§ 9 gate 5 bit-for-bit class) pass on
  the MNIST example and the sweep sources at b1000 h1024, b8192 h1024, b8192 h4096 under
  `MODELSPEC_VK_VALIDATION=1`.
- **bits-re-recorded**: the goldens, `crates/modelspec-runtime/tests/fixtures/matmul_plan_sites_same_bits.txt`,
  the recorded `train` stdout sha256 values and every checkpoint fixture are regenerated in
  this unit; the commit message lists each file and the before and after digest of the MNIST
  golden; the CHANGELOG line names ADR-009 and says the bits changed once.
- **fast-math-unchanged-in-meaning**: a `fast_math` artifact still differs from the deterministic
  one only by the `NoContraction` decorations it drops and the loss-sum and cooperative paths
  ADR-008 permits; its tolerance comparison passes unchanged.
- **throughput-reported**: per site `mean_us` and GFLOP/s of every matmul at b8192 h4096 and
  `mlp_sweep` at b8192 h4096, b8192 h1024, b1000 h1024, deterministic and `fast_math`, before
  and after, with the ratio to PyTorch eager CUDA (7,290.5, 1,932.1, 555.1 µs); raw records
  under `benchmarks/raw/<date>-fma/`; GPU alone during timed runs, clocks not locked.
- **design-states-rule**: DESIGN.md § 3.3 and § 5.4 state the fused operation and the oracle's
  `mul_add`; the revision line is bumped; § 12 ADR-009 is marked implemented with the unit's
  commit.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/spirv/emit.rs` (every accumulation site)
- `crates/modelspec-core/src/cpu/` or wherever the CPU backend multiplies and accumulates
  (matmul, reductions, loss)
- goldens, `crates/modelspec-runtime/tests/fixtures/`, checkpoint fixtures, `website/docs/guides/inspect-a-run.md`
  through `task docs-generate`
- `DESIGN.md` § 3.3, § 5.4, § 12
- `benchmarks/raw/<date>-fma/`
- not in scope: the tile rule, cooperative matrices, the parallel loss sum

## Not this story

- Any change to the order of accumulation, split-K or the two-stage loss order.
- Cooperative-matrix operands (ADR-008 amendment; story:fast-math-cooperative-matrix).
