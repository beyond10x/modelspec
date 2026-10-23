---
format: aep.planning-md/3
id: story:m1-compile-artifact
kind: story
status: implemented
title: 'modelspec compile: the CompiledArtifact on disk, with its SPIR-V modules, and a validating reader'
summary: artifact.json and kernels/<id>.spv, deterministic byte-equal compiles, all five kernel kinds valid under spirv-val, and a reader that refuses tampered artifacts with E-ARTIFACT-001 to 004
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:m1-vulkan-reduce-matmul
- depends_on: story:m1-kernel-plan
- depends_on: story:m1-mlp-example
- depends_on: story:m1-checkpoint-cpu
- depends_on: story:m1-vulkan-train
- depends_on: story:m1-vulkan-profile
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
- confidence: inferred
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-docs/src/cli.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/docs/reference/diagnostics.md
revision: 10
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T23:58:00Z", actor: "human:timo", revision: 8, decided_on: {"recorded":{"review_outcome":6}}}
- {from: "proposed", to: "active", at: "2026-10-22T23:58:00Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"review_outcome":6}}}
- {from: "active", to: "implemented", at: "2026-10-23T01:48:22Z", actor: "human:timo", revision: 10, decided_on: {"recorded":{"test_result":1,"review_outcome":7}}}
---
# `modelspec compile`: the CompiledArtifact on disk, with its SPIR-V modules, and a validating reader

## Context

The parent epic names `compile` among its CLI verbs ("`inspect-device`, `compile`, `train`, `infer`
CLI verbs"). DESIGN.md § 7.1: "The compiler emits a `CompiledArtifact { manifest, kernels, plan }`;
the runtime never reparses source", the artifact "is versioned", and "Untrusted serialized artifacts
are validated before dispatch". § 8 names `modelspec compile <file> --backend vulkan --output <dir>`
and says manifest output from `compile` lets a user inspect the generated SPIR-V. § 9 gate 8 tests
"invalid/oversized model artifacts". This story serves those three: the verb, the versioned file
set and the validating reader.

`story:m1-kernel-plan` builds the artifact in memory, and `story:m1-vulkan-reduce-matmul` owns
`spirv::emit::emit_kernels`, which returns every KernelModule of an artifact. `story:m1-vulkan-train`
compiles in memory through that entry and does not wait for this story. This story writes the
artifact and its modules to disk and reads them back, so a later benchmark or inference run can load
one file set instead of compiling again.

## What it adds

- `modelspec compile <file> --backend vulkan|cpu --output <dir> [--device-profile <json>]`.
  `--device-profile` takes a DeviceProfile JSON as `inspect-device --format json` prints it; without
  it a Vulkan compile probes device 0. A Vulkan compile runs `negotiate` (`story:m1-inspect-device`)
  on that profile before it writes anything and exits 1 with E-TARGET-004 for a profile below the
  required Vulkan version. A Cpu compile needs no device. Before it exits 0, `compile`
  reads its own output back through the reader below.
- The layout `<dir>/artifact.json` (format version 1: manifest, plan, forward plan, and one entry
  per kernel with its id, entry point, cache key, byte length and the blake3 of its bytes) and
  `<dir>/kernels/<kernel_module_id>.spv`. `artifact_hash` is blake3 over the canonical JSON with
  that field empty, followed by each module in id order; `artifact_id` derives from it. Compiles
  record `deterministic = true` and `fast_math = false`.
- A reader in `modelspec-core` that checks, before anything is dispatched, in this order: the format
  version (E-ARTIFACT-001: unreadable, or an unknown format version); that every module file exists
  and is no longer than its entry says (E-ARTIFACT-004); each module hash and then `artifact_hash`
  (E-ARTIFACT-002); the invariants of `spec/domains/artifact.yaml`, the plan verifier of
  `story:m1-kernel-plan`, that every Dispatch names a module the artifact owns, and every byte size
  computed from a shape without overflow (E-ARTIFACT-003). An altered file that is not re-sealed
  (its `artifact_hash` not recomputed) stops at 002; one that is re-sealed passes 002 and stops at
  the check it breaks.
- The four codes, declared next to the code that emits them and listed on the diagnostics page.
- It adds no crate dependency: `blake3` comes from `story:m1-kernel-plan`, `serde` and
  `serde_json` from `story:m1-inspect-device`, and the tests write under `CARGO_TARGET_TMPDIR` as
  `crates/modelspec-cli/tests/train_linear.rs` does.

## Domain relations

- CompiledArtifact owns KernelModule (`CompiledArtifact.kernels`, kind owns, cardinality many, via
  `artifact_id`, `spec/domains/artifact.yaml`): one `.spv` file per module, named by its id.
  Inferable from ess/1.
- KernelModule to DeviceProfile by value through `KernelModule.cache_key` (type `KernelCacheKey`:
  vendor_id, device_id, driver_version). Inferable from ess/1.

## Scope

- `crates/modelspec-core/src/artifact.rs` (the serialized form and the reader; `story:m1-kernel-plan`
  creates the file first).
- `crates/modelspec-cli/src/lib.rs`, `crates/modelspec-cli/src/compile.rs` (new),
  `crates/modelspec-cli/tests/compile.rs` (new), `crates/modelspec-cli/tests/compile_adversary.rs`
  (new). `story:m1-checkpoint-cpu` also edits `lib.rs`, the docs `cli.rs` and `diagnostics.rs`, and
  lands first.
- `crates/modelspec-docs/src/diagnostics.rs`, `crates/modelspec-docs/src/cli.rs`, and through
  `task docs-generate` `website/docs/reference/cli.md` and `website/docs/reference/diagnostics.md`.
- `DESIGN.md` § 7.1 and § 8 (the layout, the hash rule, the reader order, the four codes). The
  revision line, the § 8 verb table and `CHANGELOG.md` are append-only surfaces: this story adds its
  entry, and the wave coordinator assigns the revision number at merge.
- `CHANGELOG.md`.
- `website/docs/guides/inspect-a-run.md`, after `story:m1-vulkan-profile` (acceptance 7).
- Not `crates/modelspec-runtime/src/spirv/emit.rs`: `emit_kernels` is
  `story:m1-vulkan-reduce-matmul`, and this story calls it.
- Ordering in this set: this story depends on `story:m1-vulkan-train`; both edit
  `crates/modelspec-docs/src/diagnostics.rs` and regenerate `diagnostics.md` and `cli.md`, and
  vulkan-train lands first.

## Acceptance

`compile` writes an artifact whose SPIR-V passes `spirv-val` for all five kernel kinds of checkbox
5, two compiles are identical, the reader refuses every tampered artifact with the code its order
gives, and `task ci` never creates a Vulkan instance.

1. `cargo test -p modelspec-cli --features vulkan --test compile`, in `task ci`, with
   `--device-profile crates/modelspec-runtime/tests/fixtures/rtx3090.device.json`: `modelspec
   compile examples/mnist_mlp.mspec --backend vulkan --output <dir>` exits 0; `artifact.json` has one
   kernel entry per file in `kernels/`; the kernels include at least one each of elementwise,
   reduction (the `mse` loss sum), transpose, matmul and SGD update; every `kernels/*.spv` passes
   `spirv-val --target-env vulkan1.2`; the reader accepts the directory.
2. Two compiles of one file into two directories give byte-equal trees (`diff -r` empty).
3. The Cpu and Vulkan artifacts of one file have equal `schema_hash` and `parameters`; the Cpu one
   has no `kernels/` entry and no Dispatch step.
4. `cargo test -p modelspec-cli --features vulkan --test compile_adversary`: the reader refuses each
   of nine altered artifacts with the code stated, and none of them panics. Not re-sealed: one byte
   changed in a module (002); a dependency on a later step (002, the hash check comes before the
   plan verifier). Re-sealed: `format_version` 2 (001); a Dispatch naming a module not in the
   artifact (003); a dependency on a later step (003); a view past its buffer (003); an element
   count whose byte size passes `u64::MAX` (003). Either way: a module file deleted (004); one byte
   appended to a module (004, before its hash is read).
5. With `MODELSPEC_VULKAN_LIBRARY` set to a path that does not exist: `compile --backend cpu`
   exits 0; `compile --backend vulkan` without `--device-profile` exits 1 and its stderr starts with
   `E-TARGET-002: `; with `--device-profile` and the fixture it exits 0; with a copy of the fixture
   whose `subgroup_size` is 0 it exits 1 and stderr names the file and the invariant
   `subgroup_size > 0`; with a copy whose `vulkan_api_version` is 1.1.0 it exits 1, its stderr
   starts with `E-TARGET-004: `, and the output directory is not created.
6. `task ci` exits 0, `task docs-check` included, with E-ARTIFACT-001 to 004 on the diagnostics
   page and `compile` on the CLI page. Every `compile --backend vulkan` in `task ci` runs with
   `--device-profile` or with the variable of item 5, so no test in `task ci` creates a Vulkan
   instance.
7. `website/docs/guides/inspect-a-run.md`, which `story:m1-vulkan-profile` creates (wave m1-e,
   2026-10-22, took that story first), gains two fenced shell blocks with the command lines
   `modelspec compile examples/mnist_mlp.mspec --backend vulkan --output` and `spirv-dis` on one
   kernel, so the page shows all four inspection paths of DESIGN.md § 9 checkbox 9; `task website`
   exits 0 and `grep -c` of each command line on the page is at least 1.

## Out of scope

- Emitting the modules: `story:m1-vulkan-elementwise`, `story:m1-vulkan-reduce-matmul`.
- Running the artifact from disk: `story:m1-vulkan-train` compiles in memory; loading a compiled
  directory in `train` or `infer` is later work.
- A pipeline cache on disk keyed by `KernelCacheKey`: M2 measures pipeline creation first
  (`story:m1-vulkan-profile` reports it).
- Signing artifacts.
