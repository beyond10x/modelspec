---
format: aep.planning-md/3
id: story:m1-vulkan-reduce-matmul
kind: story
status: draft
title: SPIR-V reduction, transpose and naive matmul kernels, deterministic and checked against the CPU
summary: The shape-fixed reduction rule in DESIGN.md and the spec, two-stage reduce, Mse loss sum, transpose and naive matmul kernels, emit_kernels for a whole artifact, within tolerance at the MNIST shapes, matmul and loss sum bit for bit on a device with the float controls
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:m1-vulkan-elementwise
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/memory.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/common/mod.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/spirv_val.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_reduce_matmul.rs
- confidence: cited
  path: spec/domains/artifact.yaml
revision: 6
---
# SPIR-V reduction, transpose and naive matmul kernels, deterministic and checked against the CPU

## Context

DESIGN.md § 5.2 lists the remaining M1 reference kernels: "2-stage reduction (workgroup partials
then final), with numerical behavior documented", a "straightforward bounds-safe tiled or naive
`f32`" matmul "not claimed as performant", and transpose. M1 checkbox 5 of § 9 names "f32
elementwise, reduction, transpose, matmul and update kernels"; `story:m1-vulkan-elementwise`
covers elementwise and update, and this story covers the other three and the entry that emits
every kernel of an artifact, which `story:m1-vulkan-train` calls in memory and
`story:m1-compile-artifact` writes to disk.

DESIGN.md § 2.2 fixes the CPU order: a sum accumulates in row-major order from its first term, and
`matmul` sums each output element over ascending `k`, with no fused multiply-add. § 3.3 says "Do not
... reassociate floating-point reductions in deterministic mode", and
`spec/domains/artifact.yaml` summarises `CompiledArtifact.deterministic` as "no floating-point
reduction is reassociated". A two-stage reduction reassociates the sum, so § 3.3 and § 5.2 disagree
for the GPU path; this story settles it in DESIGN.md and the spec before code. M1 artifacts are
compiled with `deterministic = true`.

This story is written against DESIGN.md revision 0.9 (the 10× patch). Its § 5.4 puts a loss sum
accumulated by one invocation in row-major order and every matmul output without split-K in the
bit-for-bit class on a device that reports `shaderRoundingModeRTEFloat32`,
`shaderDenormPreserveFloat32` and `shaderSignedZeroInfNanPreserveFloat32`, and every parallel or
split reduction in the tolerance class. This story owns the reduction rule and the gate-5 tolerance
for reductions, matmul and a whole run; `story:m1-vulkan-elementwise` owns it for elementwise
kernels.

## What it adds

- DESIGN.md § 3.3, the paragraph after the pass list: in `deterministic` mode every floating-point
  reduction runs in an order fixed by the shape alone; the workgroup size is a compile-time
  constant, not read from a DeviceProfile, so one artifact gives the same bits on every run on one
  device and driver. The CPU order of § 2.2 is one admissible order.
- DESIGN.md § 5.2, M1 kernels: the two-stage order (stage 1 reduces each block of 64 consecutive
  elements in a pairwise tree in shared memory; stage 2 is one workgroup of 64 in which invocation
  `i` sums partials `i`, `i + 64`, ... in ascending order and the same tree follows; no atomics);
  the `mse` loss sum, one invocation over `(a - b) * (a - b)` in row-major order from the first
  term, then the division by the element count; matmul, one invocation per output element over
  ascending `k`, starting from the `k = 0` product and not from zero, no split-K.
- DESIGN.md § 5.4, an M1 sentence on top of 0.9: in M1 the bit-for-bit class holds the `mse` loss
  sum before its division and every matmul output; M1 emits `RoundingModeRTE`, `DenormPreserve` and
  `SignedZeroInfNanPreserve` for those two kernels on a device whose profile reports the three
  features, and asserts their bit-equality against the CPU on the operator run. Two-stage
  reductions, the `mse` value after its division and every other M1 kernel are checked within
  tolerance.
- DESIGN.md § 9 gate 5, extending the elementwise rule: for a reduction or a matmul element `r` is
  the sum of the absolute values of its terms (for `mse`, of the squared differences divided by the
  element count); after a run every parameter passes `||g - c||_2 <= 1e-4 * ||c||_2`; a
  bit-for-bit result also passes the tolerance (§ 5.4 item 3).
- `spec/domains/artifact.yaml`: the summary of `CompiledArtifact.deterministic` (line 188 before
  `story:m1-kernel-plan` adds its fields) says the same as § 3.3: every floating-point reduction
  in an order fixed by the shape alone, workgroup sizes compile-time constants, the CPU order one
  admissible order, recorded in the artifact hash. `ess specify validate --path .` runs from inside
  `spec/`.
- `crates/modelspec-runtime/src/spirv/emit.rs`: kernels for `ReduceSum` and `ReduceMean` over all
  elements (the two stages above), the `mse` loss sum (built with its final division, as the plan
  uses it, or without, for the bit-equality check), `Transpose` of rank-2 tensors, and `MatMul` with
  a bounds check on both output dimensions. Every `OpFAdd`, `OpFSub` and `OpFMul` of the loss-sum
  and matmul kernels carries `NoContraction`; when the profile reports all three float-control
  values they also declare the three execution modes for 32-bit floats, and their cache key names
  the capabilities they use.
- `spirv::emit::emit_kernels(&CompiledArtifact, &DeviceProfile)`: one KernelModule for every
  kernel of an artifact, with its entry point, SPIR-V bytes and cache key. It moves here from
  `story:m1-compile-artifact`, so `story:m1-vulkan-train` needs no artifact on disk.
- The Vulkan side in `crates/modelspec-runtime/src/vulkan/dispatch.rs` and `memory.rs`: an
  `Allocate` step makes the partials scratch buffer, and a reduction is two dispatches with a
  barrier between them; transpose, matmul and the loss sum are one dispatch each, recorded with the
  calls of `story:m1-vulkan-elementwise`.
- `crates/modelspec-runtime/tests/common/mod.rs`: the `r` of a reduction and of a matmul element,
  and the whole-run L2 check, on the constants `story:m1-vulkan-elementwise` defines.

## Domain relations

- CompiledArtifact owns KernelModule (`CompiledArtifact.kernels`, kind owns, cardinality many, via
  `artifact_id`, `spec/domains/artifact.yaml`): `emit_kernels` produces the modules an artifact
  owns, one per kernel. Inferable from ess/1.
- A KernelModule names its SPIR-V capabilities in `KernelCacheKey.capabilities`, distinct by name
  (`spec/domains/artifact.yaml`, type `KernelCacheKey`). Inferable from ess/1.

## Scope

- `spec/domains/artifact.yaml` (the `deterministic` summary); `story:m1-kernel-plan` edits the file
  first, and this story depends on it through `story:m1-vulkan-elementwise`.
- `crates/modelspec-runtime/src/spirv/emit.rs`, `crates/modelspec-runtime/src/vulkan/dispatch.rs`,
  `crates/modelspec-runtime/src/vulkan/memory.rs`.
- `crates/modelspec-runtime/tests/common/mod.rs`, `crates/modelspec-runtime/tests/spirv_val.rs`,
  `crates/modelspec-runtime/tests/vulkan_reduce_matmul.rs` (new).
- `DESIGN.md` § 3.3 (the paragraph after the pass list), § 5.2 (M1 kernels), § 5.4 (one M1
  sentence, on top of revision 0.9) and § 9 gate 5. `story:observatory-input-adjoint` edits § 3.3
  pass 4 and § 4 in wave observatory-1; this story edits neither and is scoped against `main` after
  that wave merges. The revision line and `CHANGELOG.md` are append-only surfaces: this story adds
  its entry, and the wave coordinator assigns the revision number at merge.
- `CHANGELOG.md`.

## Acceptance

Reduction, loss-sum, transpose and matmul modules, and the modules `emit_kernels` builds for a
whole artifact, are valid SPIR-V in CI; on the RTX 3090 they match the CPU interpreter within the
gate-5 tolerance at the MNIST shapes and at shapes that do not fill a workgroup, give the same bits
on every run, and the matmul outputs and the loss sum equal the CPU bit for bit when the device
reports the three float-control features.

1. First: DESIGN.md § 3.3, § 5.2, § 5.4 and § 9 gate 5 state the rule, the orders, the M1
   bit-for-bit class and the `r` and whole-run bound above, and § 3.3 no longer reads as forbidding
   the two-stage reduction of § 5.2; the `deterministic` summary in `spec/domains/artifact.yaml`
   says the same, and `ess specify validate --path .`, run from inside `spec/`, prints `valid` with
   no new warning.
2. `cargo test -p modelspec-runtime --features vulkan --test spirv_val`, in `task ci` (no Vulkan
   instance; the profile is `crates/modelspec-runtime/tests/fixtures/rtx3090.device.json`):
   `emit_kernels` on the artifacts of `examples/linear_regression.mspec`,
   `crates/modelspec-docs/models/two_layer_relu.mspec`, `examples/mnist_mlp.mspec` and the
   `mean` source of `story:m1-kernel-plan` acceptance 3 returns one module per kernel, each passes
   `spirv-val --target-env vulkan1.2`, and among them is at least one each of elementwise, two-stage
   reduction, loss sum, transpose, matmul and SGD update. Every `OpFAdd`, `OpFSub` and `OpFMul` of a
   loss-sum or matmul module carries `NoContraction`.
3. The same test: on a copy of the fixture with the three float-control values true, the loss-sum
   and matmul modules declare `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve`
   for width 32 and their cache key names the capabilities; on a copy with one of them false they
   declare none of the three and the key names none; no two-stage reduction module declares them
   in either case.
4. Operator run on the RTX 3090 (the operator starts it): `MODELSPEC_VK_VALIDATION=1 cargo test -p
   modelspec-runtime --features vulkan --test vulkan_reduce_matmul -- --ignored` passes.
   `ReduceSum`, `ReduceMean` and the `mse` loss over 1, 63, 64, 65, 1,000, 4,097 and 78,400
   elements; transpose of [1, 1], [3, 5], [100, 784] and [65, 129]; matmul at [100, 784] x
   [784, 64], [784, 100] x [100, 64], [100, 64] x [64, 10], [64, 100] x [100, 10], [100, 10] x
   [10, 64], [1, 1] x [1, 1] and [17, 33] x [33, 65]. Every element passes `GPU_ATOL + GPU_RTOL * r`
   with the `r` of acceptance 1, guard words stay unchanged, and the validation layer reports 0
   messages.
5. The same run executes every case twice and both outputs are equal bit for bit.
6. The same run reads the three float-control values from the probed profile. When all three are
   true, every matmul output element at every shape of acceptance 4 and the loss sum of every `mse`
   case (the module built without its division) equal the CPU interpreter bit for bit, and the test
   prints `class: bit-for-bit`; when one is false it prints `class: tolerance` and asserts
   acceptance 4 only. The printed class goes into the verification evidence.
7. `task ci` exits 0.

## Out of scope

- Tiled or shared-memory matmul, split-K, subgroup reductions and cooperative matrices (M2, M4).
- The float-control execution modes and bit-equality for elementwise, broadcast and SGD kernels and
  for the parameters after a step (§ 5.4 item 1): the 10× work of `epic:pytorch-10x-mnist`.
- Reductions over one axis of a higher-rank tensor: no M1 example needs one.
- Fusing the loss sum or a reduction into its producer (M2).
- Writing the modules to disk: `story:m1-compile-artifact`.
