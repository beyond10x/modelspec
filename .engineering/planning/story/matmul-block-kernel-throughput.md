---
format: aep.planning-md/3
id: story:matmul-block-kernel-throughput
kind: story
status: implemented
title: The 128x128 block kernel reaches the throughput its traffic allows
relations:
- serves: vision:modelspec
- depends_on: story:matmul-tile-from-device-invocation-limit
- depends_on: story:deterministic-fma
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T22:12:49Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-24T22:12:49Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-25T02:05:19Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# The 128x128 block kernel reaches the throughput its traffic allows

## Context

Timo, 2026-10-24: "modelspec beats python+cuda in the bigger batch benchmark". After
story:matmul-tile-from-device-invocation-limit the `mlp_sweep` step at batch 8192, hidden 4096
is 12,313.7 µs deterministic and 9,226.1 µs under `--fast-math` against 7,290.5 µs for PyTorch
eager CUDA (`benchmarks/raw/2026-10-24-tile256/padonly-sweep.jsonl`,
`padonly-fastmath-sweep.jsonl`). The forward site `[8192, 784] x [784, 4096]` runs at 5,689 µs
(9.2 TFLOP/s) deterministic and 4,608 µs (11.4 TFLOP/s) under `fast_math`; the weight-gradient
site `[784, 8192]ᵀ x [8192, 4096]` at 7,285 and 5,707 µs (`padonly-profile-b8192-h4096-r1..r3.txt`,
medians of 3 runs of 220 steps). The two are 91 % of the profiled step.

The 128×128 block of 256 invocations (`MATMUL_LARGE_BLOCK`, `crates/modelspec-core/src/ir/kernel.rs`;
the emit path in `crates/modelspec-runtime/src/spirv/emit.rs`) halves global traffic per FLOP
against the 64×128 tile and measured slower at the forward site: 6,052 against 5,774 µs
deterministic, 5,024 against 4,707 under `fast_math`; a depth-16 variant 11,604 µs
(`block-profile-b8192-h4096-r1..r3.txt`, `exp-d16-*`). Nothing in the repository says why: the
runtime reads no register count, spill size or occupancy from the driver, and the SPIR-V is only
inspected for its arithmetic order. `VK_KHR_pipeline_executable_properties` reports per pipeline
executable the statistics the driver keeps (on NVIDIA: register count, local memory bytes,
binary size, subgroup size), and the RTX 3090's driver offers the extension
(`inspect-device` lists it; verify at the start of the unit).

A bound to keep in view: a `deterministic` kernel decorates every `OpFMul` and `OpFAdd` with
`NoContraction` (`emit.rs:10`, DESIGN.md § 12), so each product and sum is one instruction and
the FP32 ceiling of the GA102 is half of its fused peak (10,496 lanes × 1 op per clock, about
17.8 TFLOP/s at 1.7 GHz, inferred from the published lane count; not measured here). The forward
site is at 52 % of that bound deterministic and at 32 % of the fused peak under `fast_math`, so
the kernel, not the ceiling, is the gap at both sites today.

## Outcome

The runtime prints, per kernel of a Vulkan plan, the pipeline executable statistics the device
reports, and a diagnosis of the block kernel at the forward site names its bound from those
statistics and one probe per hypothesis. The block kernel is then changed so that it is faster
than the 64×128 tile at the forward site with the bits of every output element unchanged (one
invocation per element, ascending `k` from the `k = 0` product, no split-K, `NoContraction`
without `fast_math`), and `compile --device-profile` and `train --backend vulkan` schedule the
plan for the profile they carry, the manifest recording the limits the plan was scheduled for.
If no variant beats the 64×128 tile, the story records the measured bound and the block stays
out of shipped plans.

## Acceptance

- **pipeline-statistics**: `train --backend vulkan --profile` prints, for every kernel of the
  plan, the statistics of `VK_KHR_pipeline_executable_properties` (name, description, value, unit
  as the driver reports them: on the RTX 3090 at least register count, local memory bytes and
  binary size), and prints `pipeline statistics: not reported` on a device without the extension;
  a device test reads the register count of a matmul kernel and checks it is positive.
- **block-diagnosed**: `benchmarks/raw/<date>-blockperf/diagnosis.md` names the block kernel's
  bound at `[8192, 784] x [784, 4096]` (registers or spills, workgroup-memory bank conflicts,
  occupancy, global load width, barrier count per `k` tile, or bounds checks in the inner loop)
  with one probe per hypothesis, each a measured variant and its figure, and the pipeline
  statistics of the 64×128 tile and the block kernel side by side.
- **block-faster**: at the forward site the block kernel's median `mean_us` over 3 profiled runs
  of 220 steps is below the 64×128 tile's by more than the run-to-run spread (5 %, measured in
  story:matmul-tile-from-device-invocation-limit), deterministic and under `fast_math`; the same
  comparison at `[784, 8192]ᵀ x [8192, 4096]` is reported either way. If no variant achieves
  it, the story's closing note records every variant and its figure and the block stays out of
  shipped plans (this criterion then counts as met by the record).
- **same-bits**: every matmul site of the MNIST step plan and the sweep plans at b1000 h1024,
  b8192 h1024 and b8192 h4096 keeps its digest in
  `crates/modelspec-runtime/tests/fixtures/matmul_plan_sites_same_bits.txt` under the plans
  scheduled for the RTX 3090 profile; `train --backend vulkan` stdout sha256 equal to the base
  commit for the four sources; the SPIR-V arithmetic section of every block variant has the
  products and sums of the 64×128 tile in the same order.
- **profile-wired**: if block-faster holds, `compile --device-profile <json>` and `train
  --backend vulkan` schedule the plan for the profile's invocation limit and workgroup memory
  (`schedule_for`), the manifest records those two limits, the artifact reader and the plan
  verifier refuse a device that reports less than the recorded limits before any dispatch
  (`E-ARTIFACT-*` naming both values), and `ir --stage kernel` without a profile, the goldens
  and `compile` without a profile are unchanged. If block-faster does not hold, no command
  changes.
- **throughput-reported**: per site `mean_us` and GFLOP/s of every matmul at b8192 h4096, and
  `mlp_sweep` at b8192 h4096, b8192 h1024 and b1000 h1024, deterministic and `fast_math`, with
  the ratio to PyTorch eager CUDA (7,290.5, 1,932.1, 555.1 µs), raw records under
  `benchmarks/raw/<date>-blockperf/`, GPU alone during every timed run, clocks not locked.
- **design-states-rule**: DESIGN.md § 3.1 states which tile each shape class takes under the
  RTX 3090 profile and why, with the figures, and § 5.2 states the profile limits the manifest
  records; the revision line is bumped.

## Scope

- `crates/modelspec-runtime/src/vulkan/` (pipeline creation with
  `VK_KHR_pipeline_executable_properties`, the statistics readout, the `--profile` printer)
- `crates/modelspec-runtime/src/spirv/emit.rs` (the block kernel's loads, staging layout,
  prefetch and inner loop)
- `crates/modelspec-core/src/ir/kernel.rs`, `schedule.rs`, `compile.rs`, `artifact.rs` (profile
  limits into the plan and the manifest; only under profile-wired)
- `crates/modelspec-cli/src/compile.rs`, `train.rs` (passing the profile; only under
  profile-wired)
- `crates/modelspec-runtime/benches/mlp_sweep.rs` (scheduling for the profile)
- `DESIGN.md` § 3.1, § 5.2, § 7.1
- `benchmarks/raw/<date>-blockperf/`
- not in scope: the loss sum, elementwise fusion, cooperative matrices, any change to the
  numerics contract (a deterministic fused multiply-add is an ADR question, recorded on the wave
  page as D15)

## Not this story

- A faster `[8192, 4096] x [4096, 10]`, `[8192, 10] x [10, 4096]ᵀ` or
  `[4096, 8192]ᵀ x [8192, 10]` site (narrow outputs; a later story).
- A parallel loss sum (story:fast-math-parallel-loss-sum) and elementwise fusion
  (story:fuse-elementwise-chains-and-gradient-sums).
