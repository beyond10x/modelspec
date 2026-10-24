---
format: aep.planning-md/3
id: story:matmul-tile-from-device-invocation-limit
kind: story
status: draft
title: 'The matmul tile takes its invocation budget from the device profile: 128x128 blocks at 256 invocations'
relations:
- serves: vision:modelspec
- depends_on: story:matmul-reads-transposed-operand
revision: 2
---
# The matmul tile takes its invocation budget from the device profile

## Context

Timo, 2026-10-24: "modelspec beats python+cuda in the bigger batch benchmark". After
story:matmul-reads-transposed-operand the `mlp_sweep` step at batch 8192, hidden 4096 is
12,008.4 µs deterministic and 9,220.4 µs under `--fast-math` against 7,290.5 µs for PyTorch
eager CUDA (`benchmarks/raw/2026-10-24-transposed/`). The two large matmul sites take 4,289 and
5,315 µs of the fast-math step (12.3 and 9.9 TFLOP/s; `benchmarks/raw/2026-10-24-fastmath/fastmath-r2-profile-b8192-h4096.txt`,
steps 0 and 17), 72 % of it, so no other site can close the gap alone.

The register-blocked tile of story:matmul-shape-aware-tiling is 64×128 at 8×8 per invocation
with 128 invocations, because `MAX_WORKGROUP_INVOCATIONS = 128`
(`crates/modelspec-core/src/ir/verify.rs:817`) is the Vulkan minimum and `MatMulTile::of`
(`crates/modelspec-core/src/ir/kernel.rs:252`) hard-codes `workgroup_size = 128`. The device
profile already records the device's limit (`DeviceProfile::max_workgroup_invocations`,
`crates/modelspec-core/src/device.rs:78`; the RTX 3090 reports 1024, `inspect-device`), and
`compile --device-profile` and `train --backend vulkan` carry a profile. A 128×128 block at 8×8
per invocation needs 256 invocations and halves the global-memory traffic per FLOP of the
64×128 block (each staged `a` fragment serves 128 columns instead of 64).

## Outcome

`MatMulTile::of` takes the profile's invocation limit beside `(m, k, n)` and, where the limit
and the shape allow, selects a 128×128 block at 8×8 per invocation with 256 invocations (two
`a` fragments and one `b` fragment staged per `k` tile, within the profile's workgroup memory);
the plan verifier checks a kernel's workgroup size against the profile's limit instead of the
constant. Every output element keeps its order: one invocation, ascending `k` from the `k = 0`
product, no split-K, `NoContraction` without `fast_math`, so the bits of every matmul site are
those of the plan before the change. A profile at the Vulkan minimum (128) yields the tiles of
today, so `ir --stage kernel` without a profile and the goldens are unchanged.

## Acceptance

- **tile-from-profile**: `MatMulTile::of` takes the invocation limit; with 128 it returns the
  tiles of today for every shape (a property test over `(m, k, n)` in `1..=10_000` compares
  against the rule before the change); with 1024 and an output of at least 128 blocks of 128
  rows and `n >= 128` it returns a 128×128 block at 8×8 with 256 invocations; the printed tile
  text, the kernel signature and the module id show the block.
- **verifier-cap-from-profile**: the plan verifier refuses a kernel whose workgroup size exceeds
  the profile's `max_workgroup_invocations` with the existing code, and accepts 256 under a
  profile that allows it; `MAX_WORKGROUP_INVOCATIONS` stays the fallback for a plan with no
  profile.
- **workgroup-memory-bounded**: the staged bytes of the 128×128 block (with `b_pad`) are within
  the profile's workgroup memory at plan time; a profile whose memory is too small for the block
  gets the next smaller tile, not a refusal.
- **same-bits**: for every matmul site of the MNIST step plan and the sweep plans at b1000 h1024,
  b8192 h1024 and b8192 h4096 under the RTX 3090 profile, the device output is bit-identical to
  the plan before the change (`crates/modelspec-runtime/tests/vulkan_matmul_plan_sites_same_bits.rs`
  against the recorded digests), and `train --backend vulkan` stdout sha256 is equal for all
  four sources.
- **default-unchanged**: `ir --stage kernel` without a device profile, the goldens, and
  `adversary_guide_kernel_lines.rs` are unchanged; a `compile` with a 128-invocation profile
  writes the same artifact hash as before.
- **throughput-reported** (reported, not gated): `mlp_sweep` medians at b8192 h4096, b8192 h1024
  and b1000 h1024 and per-site GFLOP/s of the two large sites from `--profile`, deterministic
  and `--fast-math`, before and after, beside the PyTorch eager CUDA medians of
  `benchmarks/runs.yaml`; raw records under `benchmarks/raw/<date>-tile256/`.
- **transposed-staging-padded**: the staging of a transposed operand pads its workgroup-memory
  stride (`a_stride`, `b_stride` of `Staging::of`) so consecutive invocations store into
  different banks (review-result:adversary-transposed-pass-1, finding F3: unpadded strides give
  2- to 16-way conflicts); the two weight-gradient sites' `mean_us` at b8192 h4096 are reported
  before and after beside the 427.6 and 7,229.9 µs of story:matmul-reads-transposed-operand,
  and the bits stay equal (same-bits above covers them).
- **design-states-rule**: DESIGN.md § 3.1 and § 5.2 state the block rule with the profile's
  limit and that the output order is unchanged; the revision line is bumped.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-core/src/ir/verify.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `DESIGN.md`

## Out of scope

- Split-K or any change of accumulation order.
- Cooperative matrices (story:fast-math-cooperative-matrix).
- The weight-gradient sites' in-place transposed read layout (a later story if the 128×128
  block does not lift them).
