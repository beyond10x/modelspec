# SPIR-V reduction, transpose and naive matmul kernels, deterministic and checked against the CPU

## Context

DESIGN.md § 5.2 lists the remaining M1 reference kernels: "2-stage reduction (workgroup partials
then final), with numerical behavior documented", a "straightforward bounds-safe tiled or naive
`f32`" matmul "not claimed as performant", and transpose. M1 checkbox 5 of § 9 names "f32
elementwise, reduction, transpose, matmul and update kernels"; `story:m1-vulkan-elementwise`
covers elementwise and update, and this story covers the other three.

DESIGN.md § 2.2 fixes the CPU order: a sum accumulates in row-major order from its first term, and
`matmul` sums each output element over ascending `k`, with no fused multiply-add. § 3.3 says "Do not
... reassociate floating-point reductions in deterministic mode". A two-stage reduction
reassociates the sum, so the two sections disagree for the GPU path, and this story settles it in
DESIGN.md before code. M1 artifacts are compiled with `deterministic = true`.

## What it adds

- DESIGN.md § 3.3 and § 5.2: in deterministic mode a GPU reduction uses a fixed tree that the
  compiler chooses from the shape alone (workgroup size, then partials in index order, no atomics),
  so one artifact gives the same bits on every run on one device and driver; it is not the CPU
  order, which is why gate 5 compares within tolerance. Matmul keeps the CPU order: ascending `k`,
  each product and sum decorated `NoContraction`.
- `spirv/emit.rs` kernels for `ReduceSum` and `ReduceMean` over all elements (stage 1 per-workgroup
  partials in shared memory, stage 2 one workgroup over the partials), `Mse` as squared differences
  into the same reduction, `Transpose` of rank-2 tensors, and `MatMul` with one invocation per
  output element and a bounds check on both output dimensions.
- The Vulkan dispatch of each: two dispatches and a barrier for a reduction, one for transpose and
  matmul, recorded with the calls of `story:m1-vulkan-elementwise`.

## Domain relations

None beyond `story:m1-vulkan-elementwise`: these are further KernelModules of one artifact.

## Scope

- `crates/modelspec-runtime/src/spirv/emit.rs`, `crates/modelspec-runtime/src/vulkan/dispatch.rs`.
- `crates/modelspec-runtime/tests/spirv_val.rs`,
  `crates/modelspec-runtime/tests/vulkan_reduce_matmul.rs` (new).
- `DESIGN.md` § 3.3 and § 5.2 (the deterministic reduction tree), revision line bumped.
  `CHANGELOG.md`.

## Acceptance

Reduction, transpose and matmul modules are valid SPIR-V in CI and on the RTX 3090 match the CPU
interpreter within the gate-5 tolerance, at the MNIST shapes and at shapes that do not fill a
workgroup, with the same bits on every run.

1. First: DESIGN.md § 3.3 and § 5.2 state the deterministic GPU reduction tree and the matmul order
   above, with the revision line bumped, and § 3.3 no longer reads as forbidding the two-stage
   reduction of § 5.2.
2. `cargo test -p modelspec-runtime --features vulkan --test spirv_val`, in `task ci`: every
   reduction, Mse, transpose and matmul module the emitter builds passes `spirv-val --target-env
   vulkan1.1`, and every matmul module carries `NoContraction` on each multiply and add.
3. Operator run on the RTX 3090 (the operator starts it): `MODELSPEC_VK_VALIDATION=1 cargo test -p
   modelspec-runtime --features vulkan --test vulkan_reduce_matmul -- --ignored` passes. Reductions
   and Mse over 1, 63, 64, 65, 1,000, 4,097 and 78,400 elements; transpose of [1, 1], [3, 5],
   [100, 784] and [65, 129]; matmul at [100, 784] x [784, 64], [784, 100] x [100, 64],
   [100, 64] x [64, 10], [64, 100] x [100, 10], [100, 10] x [10, 64], [1, 1] x [1, 1] and
   [17, 33] x [33, 65]. Every element passes `GPU_ATOL + GPU_RTOL * r` of
   `crates/modelspec-runtime/tests/common/mod.rs`, guard words stay unchanged, and the validation
   layer reports 0 messages.
4. The same run executes every case twice and both outputs are equal bit for bit. The count of
   matmul elements bit-equal to the CPU is printed per shape and recorded in the verification
   evidence, not asserted.
5. `task ci` exits 0.

## Out of scope

- Tiled or shared-memory matmul, subgroup reductions and cooperative matrices (M2, M4).
- Reductions over one axis of a higher-rank tensor: no M1 example needs one.
- Fusing `Mse` or a reduction into its producer (M2).
