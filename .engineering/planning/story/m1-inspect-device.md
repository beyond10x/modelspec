---
format: aep.planning-md/3
id: story:m1-inspect-device
kind: story
status: draft
title: 'Vulkan device discovery and feature negotiation: modelspec inspect-device --backend vulkan'
summary: A compute-only Vulkan 1.2 device through a run-time loader, DeviceProfile with the three float-control fields probed and printed as text or JSON, negotiation in one pure function, E-TARGET-002 to 004, and an RTX 3090 profile fixture
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:observatory-live-event-stream
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: Cargo.toml
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/Cargo.toml
- confidence: inferred
  path: crates/modelspec-cli/src/inspect_device.rs
- confidence: cited
  path: crates/modelspec-cli/src/lib.rs
- confidence: cited
  path: crates/modelspec-cli/src/main.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/inspect_device.rs
- confidence: cited
  path: crates/modelspec-core/Cargo.toml
- confidence: inferred
  path: crates/modelspec-core/src/device.rs
- confidence: cited
  path: crates/modelspec-core/src/lib.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-docs/src/status.rs
- confidence: cited
  path: crates/modelspec-runtime/Cargo.toml
- confidence: cited
  path: crates/modelspec-runtime/src/lib.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan
- confidence: inferred
  path: crates/modelspec-runtime/tests/fixtures/rtx3090.device.json
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_device.rs
- confidence: cited
  path: spec/domains/device.yaml
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/docs/reference/diagnostics.md
revision: 6
---
# Vulkan device discovery and feature negotiation: `modelspec inspect-device --backend vulkan`

## Context

M1 checkbox 4 of DESIGN.md § 9 reads "Runtime finds RTX 3090 through Vulkan and prints verified
device features", and the parent epic promises a headless runtime with "device discovery, feature
negotiation, allocator, dispatch, sync". DESIGN.md § 5.1 lists what discovery records: vendor and
device ids, the device UUID if available, driver version, SPIR-V and Vulkan feature levels,
subgroup properties, f16 storage and compute features, cooperative-matrix extensions and
properties, workgroup and storage-buffer limits, timestamp support, and memory heaps and budgets.
DESIGN.md § 8 adds that `inspect-device` reports each capability as the driver states it, "not
infer it from `RTX 3090` as a product name". DESIGN.md § 5.4 (revision 0.9, the 10× patch this
story is written against) puts a result in the bit-for-bit class only on a device that reports
`shaderRoundingModeRTEFloat32`, `shaderDenormPreserveFloat32` and
`shaderSignedZeroInfNanPreserveFloat32`, so the profile must carry those three.
`spec/domains/device.yaml` types the record as `modelspec.device.DeviceProfile` (lifecycle
`Probed`); it has no UNMAPPED marker.

No Vulkan code exists: the `vulkan` feature of `crates/modelspec-runtime/Cargo.toml` is empty and
`crates/modelspec-core/tests/no_vulkan.rs` keeps every Vulkan crate out of `modelspec-core`. This
story is the first Vulkan code and the device layer every later M1 story uses.

## What it adds

- `spec/domains/device.yaml`: three boolean fields on `DeviceProfile`,
  `shader_rounding_mode_rte_f32`, `shader_denorm_preserve_f32` and
  `shader_signed_zero_inf_nan_preserve_f32`, each summarised as the driver value of the Vulkan
  property of the same name; `ess specify validate --path .` from inside `spec/` passes.
- `modelspec_core::device::DeviceProfile`, the plain data type of `spec/domains/device.yaml` with
  its invariants, and its JSON form (`serde`, `serde_json` in `modelspec-core`). No Vulkan type
  appears in it.
- `crates/modelspec-runtime/src/vulkan/` behind the `vulkan` feature: an `ash` instance with the
  loader opened at run time (no link-time dependency on `libvulkan`), physical-device enumeration,
  selection of a compute-capable queue family, a compute-only logical device (no surface, no
  swapchain), and `probe() -> DeviceProfile` filled from the Vulkan property and feature chains,
  the three float-control values from `VkPhysicalDeviceFloatControlsProperties`.
- Feature negotiation, in one place (`vulkan/negotiate.rs`): the runtime requires Vulkan API
  version 1.2 (float controls and SPIR-V 1.5 are core there) and a compute queue family, and
  enables no optional feature bit and no device extension in M1. `negotiate` is a pure function
  of a DeviceProfile; the logical device is created with exactly the set it returns. A device below
  the required version is refused with E-TARGET-004, whose message names the required and the
  reported version. f16, cooperative matrix and the float-control values are recorded and not
  required: a device lacking a float-control value runs in the tolerance class of § 5.4.
- `modelspec inspect-device --backend vulkan [--device <index>] [--format text|json]`: lists every
  physical device, or with `--device` one of them, and prints the profile and the negotiation
  verdict. `modelspec-cli` gains a `vulkan` feature, on by default, that turns on the runtime one;
  without it the verb is refused with E-TARGET-001 as `train --backend vulkan` is now.
- E-TARGET-002 (no Vulkan loader could be opened), E-TARGET-003 (no physical device, or the
  chosen device has no compute queue) and E-TARGET-004 (the device is below the required Vulkan
  version), declared next to the code that emits them and listed on the diagnostics page.
  `MODELSPEC_VULKAN_LIBRARY` names the loader to open, so a test can point it at a path that does
  not exist.
- `crates/modelspec-runtime/tests/fixtures/rtx3090.device.json`, the profile from the operator run
  of acceptance item 4, which later stories compile against without a GPU.
- DESIGN.md § 6 "Device": the required version and the enabled set. § 8: the verb, its flags and
  the three codes.

## Domain relations

- DeviceProfile is written by `probe()` and read by `compile` through `KernelCacheKey`
  (`spec/domains/artifact.yaml`, its vendor_id, device_id and driver_version fields); the key
  carries the values, not a reference to a stored profile. Inferable from ess/1.

## Scope

- `spec/domains/device.yaml` (the three float-control fields).
- `Cargo.toml` (workspace dependencies: `ash` with its `loaded` feature, `serde`, `serde_json`),
  `Cargo.lock`. Root `Cargo.toml` is also edited by `story:observatory-live-event-stream` (wave
  observatory-1); this story is scoped against `main` after that wave merges.
- `crates/modelspec-core/Cargo.toml` (`serde`, `serde_json`), `crates/modelspec-core/src/device.rs`
  (new), `crates/modelspec-core/src/lib.rs`.
- `crates/modelspec-runtime/Cargo.toml` (the `vulkan` feature and its dependency; the bench entry
  of `story:cpu-matmul-row-order` is the only other edit to this file, and that story merges
  first), `crates/modelspec-runtime/src/lib.rs`, `crates/modelspec-runtime/src/vulkan/` (new).
- `crates/modelspec-runtime/tests/vulkan_device.rs` (new),
  `crates/modelspec-runtime/tests/fixtures/rtx3090.device.json` (new, committed after the operator
  run).
- `crates/modelspec-cli/Cargo.toml`, `crates/modelspec-cli/src/main.rs`,
  `crates/modelspec-cli/src/lib.rs`, `crates/modelspec-cli/src/inspect_device.rs` (new),
  `crates/modelspec-cli/tests/inspect_device.rs` (new).
- `crates/modelspec-docs/src/diagnostics.rs`, `crates/modelspec-docs/src/status.rs`, and through
  `task docs-generate` `website/docs/reference/cli.md` and `website/docs/reference/diagnostics.md`.
  `story:observatory-live-event-stream` also regenerates `cli.md`; this story regenerates it on
  `main` after that wave merges.
- `DESIGN.md` § 6 "Device" and § 8. The revision line, the § 8 verb table and `CHANGELOG.md` are
  append-only surfaces: this story adds its entry, and the wave coordinator assigns the revision
  number at merge.
- `AGENTS.md` (State), `CHANGELOG.md`.

## Acceptance

On the RTX 3090 `inspect-device` prints every DeviceProfile field as the driver reports it and a
passing negotiation verdict, the JSON parses back into a profile that passes its invariants, a
device below Vulkan 1.2 is refused with a stated code, and with no loader the verb fails with a
stated code instead of crashing.

Order: the operator run of item 4 produces `rtx3090.device.json`; the coordinator commits it to
the story branch; items 1, 3 and 7, which read it, are checked after that commit. The story is
done only after the operator run.

1. `cargo test -p modelspec-core device`: a DeviceProfile read from
   `crates/modelspec-runtime/tests/fixtures/rtx3090.device.json` writes back byte for byte, and
   each invariant of `spec/domains/device.yaml` (`subgroup_size > 0`, `max_workgroup_invocations >
   0`, `max_shared_memory_bytes > 0`, distinct stages and operations, no cooperative-matrix shapes
   when `cooperative_matrix` is false) refuses a profile that breaks it, one test per invariant.
2. `cargo test -p modelspec-cli --test inspect_device`, in `task ci` and on a host without a GPU:
   with `MODELSPEC_VULKAN_LIBRARY` set to a path that does not exist, `modelspec inspect-device
   --backend vulkan` exits 1 and its stderr starts with `E-TARGET-002: `; `--backend cpu` exits 2;
   `--format yaml` exits 2. No test in `task ci` creates a Vulkan instance.
3. `cargo test -p modelspec-runtime --features vulkan negotiate`, in `task ci` (no instance): on
   the fixture `negotiate` returns the required set (API 1.2, no optional feature, no extension);
   on a copy with `vulkan_api_version` 1.1.0 it returns E-TARGET-004 and the message names 1.2.0
   and 1.1.0; on a copy with one float-control value false it passes and reports the tolerance
   class.
4. Operator run on the RTX 3090, started by the operator:
   `cargo test -p modelspec-runtime --features vulkan --test vulkan_device -- --ignored` passes and
   `modelspec inspect-device --backend vulkan --device 0 --format json` exits 0 with
   `vendor_id` 4318 (0x10de), a non-zero `device_id` and `driver_version`, `vulkan_api_version`
   at least 1.2.0, `subgroup_size` 32, `max_workgroup_invocations` at least 1024,
   `max_shared_memory_bytes` at least 49152, at least one `memory_heaps` entry with `device_local`
   true, and `cooperative_matrix` and the three float-control fields as the driver reports them
   (recorded, not asserted). The device test asserts that the logical device was created with the
   set `negotiate` returned. Its JSON output is the fixture the coordinator commits.
5. The same run with `--format text` prints one line per DeviceProfile field of
   `spec/domains/device.yaml`, the 24 fields besides the identity `device_profile_id`, each as
   `<field>: <value>`, then `negotiation: ok`.
6. `ess specify validate --path .` from inside `spec/` exits 0 with the three new fields.
7. `task ci` exits 0, `task docs-check` included, with E-TARGET-002, E-TARGET-003 and E-TARGET-004
   on the diagnostics page.

## Out of scope

- Memory allocation, pipelines and dispatch: `story:m1-vulkan-elementwise`.
- Choosing a kernel by capability (cooperative matrix, f16): M2 and M4.
- Emitting the float-control execution modes: `story:m1-vulkan-reduce-matmul`.
- Lavapipe or any software Vulkan device in CI.
- Memory budgets through `VK_EXT_memory_budget` beyond recording `budget_bytes` when the extension
  is present.
