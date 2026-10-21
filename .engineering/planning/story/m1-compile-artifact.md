---
format: aep.planning-md/3
id: story:m1-compile-artifact
kind: story
status: draft
title: 'modelspec compile: the CompiledArtifact on disk, with its SPIR-V modules, and a validating reader'
summary: artifact.json and kernels/<id>.spv, deterministic byte-equal compiles, all five kernel kinds valid under spirv-val, and a reader that refuses tampered artifacts with E-ARTIFACT-001 to 004
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:m1-vulkan-reduce-matmul
- depends_on: story:m1-kernel-plan
- depends_on: story:m1-mlp-example
- depends_on: story:m1-checkpoint-cpu
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/src/compile.rs
- confidence: cited
  path: crates/modelspec-cli/src/lib.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/compile.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/compile_adversary.rs
- confidence: cited
  path: crates/modelspec-core/Cargo.toml
- confidence: inferred
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-docs/src/cli.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: website/docs/reference
revision: 3
---
# `modelspec compile`: the CompiledArtifact on disk, with its SPIR-V modules, and a validating reader

## Context

DESIGN.md § 7.1: "The compiler emits a `CompiledArtifact { manifest, kernels, plan }`; the runtime
never reparses source", the artifact "is versioned", and "Untrusted serialized artifacts are
validated before dispatch". § 8 names `modelspec compile <file> --backend vulkan --output <dir>`
and says manifest output from `compile` lets a user inspect the generated SPIR-V. § 9 gate 8 tests
"invalid/oversized model artifacts". M1 checkbox 5 asks that the compiler emit valid SPIR-V for
f32 elementwise, reduction, transpose, matmul and update kernels.

`story:m1-kernel-plan` builds the artifact in memory, and `story:m1-vulkan-elementwise` and
`story:m1-vulkan-reduce-matmul` emit every M1 kernel. This story writes and reads the artifact, so
training, inference and later benchmark runs load one file set instead of compiling again.

## What it adds

- `modelspec compile <file> --backend vulkan|cpu --output <dir> [--device-profile <json>]`.
  `--device-profile` takes a DeviceProfile JSON as `inspect-device --format json` prints it; without
  it a Vulkan compile probes device 0. A Cpu compile needs no device.
- The layout `<dir>/artifact.json` (format version 1: manifest, plan, forward plan, and one entry
  per kernel with its id, entry point, cache key and the blake3 of its bytes) and
  `<dir>/kernels/<kernel_module_id>.spv`. `artifact_hash` is blake3 over the canonical JSON with
  that field empty, followed by each module in id order; `artifact_id` derives from it. Compiles
  record `deterministic = true` and `fast_math = false`.
- A reader in `modelspec-core` that checks, before anything is dispatched: the format version,
  `artifact_hash`, each module hash, the invariants of `spec/domains/artifact.yaml`, the plan
  verifier of `story:m1-kernel-plan`, that every Dispatch names a module the artifact owns, and
  sizes against the file sizes. It refuses with E-ARTIFACT-001 (unreadable or unknown format
  version), E-ARTIFACT-002 (a hash does not match), E-ARTIFACT-003 (an invariant or the plan
  verifier fails) and E-ARTIFACT-004 (a module is missing or larger than its entry says).

## Domain relations

- CompiledArtifact owns KernelModule (`kernels`, `spec/domains/artifact.yaml:213`): one `.spv`
  file per module, named by its id. Inferable from ess/1.
- KernelModule to DeviceProfile by value through `KernelCacheKey` (line 146). Inferable from ess/1.

## Scope

- `crates/modelspec-core/src/artifact.rs`, `crates/modelspec-core/Cargo.toml`.
- `crates/modelspec-runtime/src/spirv/emit.rs` (the entry that emits every kernel of an artifact).
- `crates/modelspec-cli/src/lib.rs`, `crates/modelspec-cli/src/compile.rs` (new),
  `crates/modelspec-cli/tests/compile.rs` (new), `crates/modelspec-cli/tests/compile_adversary.rs`
  (new).
- `crates/modelspec-docs/src/diagnostics.rs`, `crates/modelspec-docs/src/cli.rs`,
  `website/docs/reference/` through `task docs-generate`.
- `DESIGN.md` § 7.1 and § 8 (the layout, the hash rule, the four codes), revision line bumped.
  `CHANGELOG.md`.

## Acceptance

`compile` writes an artifact whose SPIR-V passes `spirv-val` for all five kernel kinds of checkbox
5, two compiles are identical, and the reader refuses every tampered artifact with a stated code.

1. `cargo test -p modelspec-cli --features vulkan --test compile`, in `task ci`, with
   `--device-profile crates/modelspec-runtime/tests/fixtures/rtx3090.device.json`: `modelspec
   compile examples/mnist_mlp.mspec --backend vulkan --output <tmp>` exits 0; the kernels include
   at least one each of elementwise, reduction, transpose, matmul and SGD update; every
   `kernels/*.spv` passes `spirv-val --target-env vulkan1.1`.
2. Two compiles of one file into two directories give byte-equal trees (`diff -r` empty).
3. The Cpu and Vulkan artifacts of one file have equal `schema_hash` and `parameters`; the Cpu one
   has no `kernels/` entry and no Dispatch step.
4. `cargo test -p modelspec-cli --features vulkan --test compile_adversary`: the reader refuses
   each of seven altered artifacts, with the code stated: one byte changed in a module (002), a
   module deleted (004), `format_version` 2 (001), a Dispatch naming a module not in the artifact
   (003), a dependency on a later step (003), a view past its buffer (003), and an element count
   whose byte size passes `u64::MAX` (003). None of them panics.
5. `compile --backend cpu` without `--device-profile` exits 0 on a host without Vulkan;
   `--backend vulkan` there exits 1 with E-TARGET-002; a profile file that breaks an invariant exits
   1 naming the invariant.
6. `task ci` exits 0 with E-ARTIFACT-001 to 004 on the diagnostics page.

## Out of scope

- Running the artifact: `story:m1-vulkan-train`.
- A pipeline cache on disk keyed by `KernelCacheKey`: M2 measures pipeline creation first
  (`story:m1-vulkan-profile` reports it).
- Signing artifacts.
