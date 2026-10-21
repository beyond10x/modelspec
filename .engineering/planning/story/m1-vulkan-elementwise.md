---
format: aep.planning-md/3
id: story:m1-vulkan-elementwise
kind: story
status: draft
title: Vulkan memory, pipelines and dispatch, with SPIR-V elementwise, broadcast and SGD kernels
summary: The M1 GPU tolerances in DESIGN.md, rspirv elementwise and SGD kernels, spirv-val in CI, buffers, pipelines, barriers and fences with record, submit and wait apart
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:m1-inspect-device
- depends_on: story:m1-kernel-plan
- depends_on: story:m1-mlp-example
scope:
- confidence: cited
  path: .github/workflows/check.yml
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Cargo.toml
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: Taskfile.yml
- confidence: cited
  path: crates/modelspec-runtime/Cargo.toml
- confidence: cited
  path: crates/modelspec-runtime/src/lib.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan
- confidence: inferred
  path: crates/modelspec-runtime/tests/common/mod.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/spirv_val.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_elementwise.rs
revision: 3
---
# Vulkan memory, pipelines and dispatch, with SPIR-V elementwise, broadcast and SGD kernels

## Context

DESIGN.md § 5.1 has the compiler build SPIR-V with `rspirv` and no GLSL or HLSL compiler in the
production path; § 5.2 lists the M1 elementwise kernels ("flat-index global dispatch, explicit
bounds checks, `f32` loads/stores") and SGD (`p_out = p_in - lr * gradient`). § 6 gives the runtime
its memory and execution duties and forbids "hidden host synchronization between every operator".
§ 9 gate 5 compares Vulkan and CPU "within documented tolerances", but no tolerance is documented
yet; `story:bert-vulkan-kernels-rowwise` uses `|g - c| <= 1e-5 + 1e-4 * |c|` and says it follows
M1 if M1 documents others. Gate 6 runs `spirv-val` on every generated module in CI and validation
layers in GPU tests.

`story:m1-inspect-device` brings the device; `story:m1-kernel-plan` brings the kernels and plan
this story turns into SPIR-V and dispatches.

## What it adds

- The M1 tolerances, in DESIGN.md § 9 gate 5 and as `GPU_ATOL = 1e-5` and `GPU_RTOL = 1e-4` in
  `crates/modelspec-runtime/tests/common/mod.rs`: an element passes when `|g - c| <= GPU_ATOL +
  GPU_RTOL * r`, where `r` is `|c|` for elementwise kernels and the sum of the absolute values of
  the terms for a reduction or a matmul element; after a run every parameter passes `||g - c||_2 <=
  1e-4 * ||c||_2`.
- `crates/modelspec-runtime/src/spirv/emit.rs`: one compute module per Kernel IR elementwise kernel
  (add, sub, mul, each also with a rank-0 operand; relu; relu-grad; scalar broadcast; SGD update),
  flat index, a bounds check against an element count, f32 only, workgroup size 64.
  `crates/modelspec-runtime/src/spirv/validate.rs` runs `spirv-val --target-env vulkan1.1` on a
  module and returns its verdict.
- `crates/modelspec-runtime/src/vulkan/{memory,pipeline,dispatch,sync}.rs` behind the `vulkan`
  feature: device-local buffers and staging buffers aligned to the profile limits, typed
  allocation errors instead of panics, a compute pipeline per module, descriptor sets and push
  constants, command recording with compute-to-compute and transfer barriers, and a fence. Record,
  submit and wait are three separate calls, so a caller can record many dispatches and wait once.
- `spirv-val` in CI: `.github/workflows/check.yml` installs `spirv-tools`; `task test` and `task
  lint` also build and test `modelspec-runtime` with `--features vulkan` (no test in `task ci`
  creates a Vulkan instance).
- `MODELSPEC_VK_VALIDATION=1` turns on the Khronos validation layer, and any validation message
  fails the device test that caused it.

## Domain relations

- A KernelModule carries its `cache_key` with the vendor, device and driver of the profile it was
  built for (`spec/domains/artifact.yaml:146`). Inferable from ess/1.

## Scope

- `Cargo.toml` (workspace dependency `rspirv`), `crates/modelspec-runtime/Cargo.toml`,
  `crates/modelspec-runtime/src/lib.rs`, `crates/modelspec-runtime/src/spirv/` (new),
  `crates/modelspec-runtime/src/vulkan/`.
- `crates/modelspec-runtime/tests/common/mod.rs` (new), `crates/modelspec-runtime/tests/spirv_val.rs`
  (new), `crates/modelspec-runtime/tests/vulkan_elementwise.rs` (new).
- `Taskfile.yml` (`test`, `lint`), `.github/workflows/check.yml`.
- `DESIGN.md` § 9 gate 5 (the tolerances), revision line bumped. `AGENTS.md` (Commands: the
  `--features vulkan` runs and `MODELSPEC_VK_VALIDATION`), `CHANGELOG.md`.

## Acceptance

Every elementwise, broadcast and SGD kernel is valid SPIR-V in CI and, on the RTX 3090, matches the
CPU interpreter within the documented tolerance on sizes that do not fill a workgroup, with one
host wait per test case.

1. First: DESIGN.md § 9 gate 5 states `GPU_ATOL = 1e-5`, `GPU_RTOL = 1e-4`, the reference `r` per
   kernel kind and the end-of-run bound `1e-4` relative L2 per parameter, with the revision line
   bumped; `crates/modelspec-runtime/tests/common/mod.rs` defines both constants and nothing else
   in the tests repeats the numbers.
2. `cargo test -p modelspec-runtime --features vulkan --test spirv_val`, in `task ci`: every module
   the emitter builds for the kernels of `examples/linear_regression.mspec` and
   `crates/modelspec-docs/models/two_layer_relu.mspec` passes `spirv-val --target-env vulkan1.1`,
   and a module with one instruction removed fails it. With `spirv-val` absent the test fails; it
   does not skip.
3. `.github/workflows/check.yml` installs `spirv-tools`, and its `check` job is green on the
   change with acceptance 2 among the tests it ran.
4. Operator run on the RTX 3090 (the operator starts it): `MODELSPEC_VK_VALIDATION=1 cargo test -p
   modelspec-runtime --features vulkan --test vulkan_elementwise -- --ignored` passes. For every
   kernel and the element counts 1, 63, 64, 65, 4,097 and 78,400 from a fixed seed, every element
   passes the tolerance of acceptance 1 against the CPU interpreter, 16 guard words after each
   output stay unchanged, and the validation layer reports 0 messages.
5. The same run chains add, relu and SGD of one case in one command buffer and the device counter
   of host waits reads 1 for the case; a test that waits per dispatch would read 3.
6. `task ci` and `task core-no-vulkan` exit 0; `rspirv` and `ash` appear under
   `modelspec-runtime` only.

## Out of scope

- Reductions, transpose and matmul: `story:m1-vulkan-reduce-matmul`.
- The artifact on disk and `compile`: `story:m1-compile-artifact`.
- A whole training run on the GPU: `story:m1-vulkan-train`.
- Vectorized loads, fusion and tuned workgroup sizes (M2).
