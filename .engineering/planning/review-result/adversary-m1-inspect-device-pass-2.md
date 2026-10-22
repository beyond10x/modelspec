---
format: aep.planning-md/3
id: review-result:adversary-m1-inspect-device-pass-2
kind: review-result
status: active
title: Adversary pass 2, story:m1-inspect-device
relations:
- reviews: story:m1-inspect-device
revision: 1
---
unit: story:m1-inspect-device, working tree of impl/m1-inspect-device (base 3a62a7a, uncommitted unit plus my three untracked files)
verdict: CONFIRMED
cases: executed 665→670, red 4
origin: introduced 7 / pre-existing 0 / undecided 0
wrote-outside-worktree: 9 paths (part 6)
needs-coordinator: none

## 1. Diff: what I touched

No tracked file changed. Three untracked test files added; every other path in `git status` was already there before this pass:

| File | Kind |
|---|---|
| crates/modelspec-cli/tests/adversary2_inspect_pre_1_2.rs | test, 4 cases (1 control) |
| crates/modelspec-runtime/tests/adversary2_create_device_old.rs | test, 1 case |
| crates/modelspec-runtime/tests/adversary2_fake_loader.c | test fixture: a stand-in loader, built at test time with `cc` into `CARGO_TARGET_TMPDIR` |

The stand-in loader includes `<vulkan/vulkan_core.h>`, so struct layouts are the real ones. It has three devices: 0 at 1.2.0, 1 at 1.1.0, 2 at 1.0.0. Each has real compute limits, one 1 GiB device-local heap and one compute queue family. Each reports subgroupSize 32 through `VkPhysicalDeviceSubgroupProperties` and `VkPhysicalDeviceVulkan11Properties`, and a UUID through `VkPhysicalDeviceIDProperties`. `vkGetPhysicalDeviceProperties2` walks pNext. `vkCreateDevice`, `vkGetDeviceProcAddr` and `vkDestroyDevice` are stubbed.

## 2. Cases added (each run alone before the suite)

| Case | Asserts | Now |
|---|---|---|
| adversary2_control_the_vulkan_1_2_device_profile_reads_back | `--device 0 --format json` exits 0 and the output passes `DeviceProfile::from_json` | green (control) |
| adversary2_a_vulkan_1_1_device_profile_keeps_the_spec_invariants | `--device 1 --format json`: exit 1, stderr `E-TARGET-004: `, and stdout passes `from_json` (spec invariants) | red |
| adversary2_a_vulkan_1_1_device_does_not_claim_spirv_1_5 | the 1.1 device's `spirv_version` is at most 0x10300 | red |
| adversary2_a_vulkan_1_0_device_does_not_claim_spirv_1_5 | the 1.0 device's `spirv_version` == 0x10000 | red |
| adversary2_create_device_refuses_or_reports_a_vulkan_1_1_device_truthfully | `create_device(1, &negotiate(device 0))` fails with E-TARGET-004, or the device reports an api_version at most 1.1 | red |

Red output, verbatim, captured from runs of these cases alone (`b10x-build-slot cargo test -p modelspec-cli --test adversary2_inspect_pre_1_2 [filter]`; `... -p modelspec-runtime --features vulkan --test adversary2_create_device_old`):

```
thread 'adversary2_a_vulkan_1_1_device_profile_keeps_the_spec_invariants' panicked at crates/modelspec-cli/tests/adversary2_inspect_pre_1_2.rs:78:5:
Err("DeviceProfile breaks invariant subgroup_size > 0 (got 0)")
```
```
thread 'adversary2_a_vulkan_1_0_device_does_not_claim_spirv_1_5' panicked at crates/modelspec-cli/tests/adversary2_inspect_pre_1_2.rs:111:5:
assertion `left == right` failed: a Vulkan 1.0 device is reported as accepting SPIR-V 1.5
device_uuid: null
subgroup_size: 0
subgroup_stages: []
subgroup_operations: []
...
vulkan_api_version: 4194304
spirv_version: 66816
...
negotiation: refused (E-TARGET-004)
  left: 66816
 right: 65536
test result: FAILED. 1 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out
```
(The 1.1 SPIR-V case failed in the same run, at `vulkan_api_version: 4198400`, `spirv_version: 66816`.)
```
thread 'adversary2_create_device_refuses_or_reports_a_vulkan_1_1_device_truthfully' panicked at crates/modelspec-runtime/tests/adversary2_create_device_old.rs:51:23:
a logical device was created on the Vulkan 1.1 device 1 and reports api_version 0x402000 (1.2), above the 1.1 the device runs at
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
```

## 3. Suite, after the cases existed

| Command | Exit | Result |
|---|---|---|
| `b10x-build-slot task ci` | 201 | stops at the first failing test binary: `adversary2_inspect_pre_1_2` `test result: FAILED. 1 passed; 3 failed` / `task: Failed to run task "test": exit status 101` |
| `b10x-build-slot cargo test --workspace --no-fail-fast` | 101 | 666 passed, 4 failed (the four above), 1 ignored. The implementor's count was 665 passed, 1 ignored, so 665→670 executed. |
| `b10x-build-slot cargo test -p modelspec-runtime --features vulkan --no-fail-fast` | 101 | 229 passed, 1 failed (adversary2_create_device_…), 1 ignored |
| `cargo fmt --check` / `cargo clippy --workspace --all-targets -D warnings` / clippy runtime `--features vulkan` | 0 / 0 / 0 | my files are lint-clean |

The pass-1 cases in `adversary_inspect_old_device.rs` are green now (included in the 666).

## 4. Findings

| # | file:line | Verdict / origin | What was measured | What reaches it |
|---|---|---|---|---|
| 1 | crates/modelspec-runtime/src/vulkan/probe.rs:239 | CONFIRMED / introduced | Below 1.2, `subgroup_size` and the subgroup stages and operations stay at the Vulkan11 defaults (0, [], []). The printed JSON fails `DeviceProfile::from_json` (`subgroup_size > 0`), although the driver reports them through `VkPhysicalDeviceSubgroupProperties`, which is core in 1.1. Red: case 2. | `inspect-device` on any host whose driver reports 1.1 (the listing prints every device). It is not shown on this host: the only real device is the RTX 3090 at 1.4. The stand-in loader builds the state. |
| 2 | crates/modelspec-runtime/src/vulkan/probe.rs:255 | CONFIRMED / introduced | `spirv_version()` maps everything below 1.3 to SPIR-V 1.5. A 1.1 device (SPIR-V 1.3 at most) and a 1.0 device (1.0) are reported as accepting 1.5: a false capability claim against device.yaml "Highest SPIR-V version the device accepts". Red: cases 3 and 4. Fix: 1.0 maps to 0x10000, 1.1 to 0x10300. | Same path as 1. |
| 3 | crates/modelspec-runtime/src/vulkan/mod.rs:100 | INFEASIBLE / introduced | `create_device` never checks the target device against `negotiate`. Given a `Negotiated` from device 0, it creates a device on the 1.1 device 1, and `api_version()` reports 1.2 (the instance value) for a device that runs at 1.1. DESIGN § 6 says the runtime "refuses a device below it". Red: case 5. | Only the ignored device test calls `create_device`. A wrong `Negotiated` has to be built on purpose (its fields are public). Later M1 stories inherit the gap. |
| 4 | crates/modelspec-runtime/src/vulkan/probe.rs:237 | CONFIRMED / introduced | Below 1.2, `device_uuid` is null and `f16_storage` and the three float-control fields are `false`. device.yaml calls the latter "the driver value", but the driver stated nothing. A 1.1 driver does expose the UUID (`VkPhysicalDeviceIDProperties`) and 16-bit storage. Visible in the case-4 output. | Same path as 1. The schema has no way to say "not reported" for the Booleans. |
| 5 | crates/modelspec-runtime/src/vulkan/enabled.rs:128 | INFEASIBLE / introduced | `of()` reads only `pEnabledFeatures` and ignores pNext. Features enabled through `VkPhysicalDeviceFeatures2`, or the 1.1/1.2 feature structs (where `pEnabledFeatures` must be NULL), would read back as empty. It also reads back the struct `create_device` just built from `negotiated`, so the device test's equality cannot diverge. | Nothing: `create_device` chains no pNext, and `assert!(negotiated.features.is_empty())` panics first. |
| 6 | crates/modelspec-runtime/src/vulkan/negotiate.rs:63 | CONFIRMED / introduced | The E-TARGET-004 line names no device index. In `--format json` listing mode the array carries no verdict either, so with two refused devices stderr shows two lines and the reader cannot tell which device each belongs to. | `inspect-device` listing on a host with a pre-1.2 device (same as 1). Judgement, no test. |
| 7 | crates/modelspec-runtime/src/vulkan/probe.rs:106 | INFEASIBLE / introduced | `spirv_version(properties.api_version)` is passed the unmasked version. Variant bits (bits 29–31) make any variant-1 device compare at or above 1.3, so it is reported as SPIR-V 1.6. `negotiate` and `current` mask correctly. | No desktop loader reports a non-zero variant. |

The findings cover the working tree of impl/m1-inspect-device as of this pass.

## 5. Attacked, not broken

- 1.0 memory path: `vkGetPhysicalDeviceMemoryProperties` is used and the heap is reported correctly.
- 1.1 memory path: `MemoryProperties2` is used, and the budget is chained only when the extension exists.
- `--device 0` on a 1.2 device beside older devices: exit 0. `--device 1`/`2`: profile printed, exit 1, stderr starts `E-TARGET-004: `.
- Listing text for refused devices: `negotiation: refused (E-TARGET-004)` per device.
- `queue_family` agrees with the probed `compute_queue_family`: both come from the same first-COMPUTE-family rule.
- `extension_name_as_c_str` is bounded. The `enabled::of` raw reads stay inside the lifetimes `create_device` holds.

## 6. Paths written outside the worktree

- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adversary-2.md
- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adv2-red-cli.log
- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adv2-red-cli-invariant.log
- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adv2-red-runtime.log
- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adv2-ci.log
- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adv2-suite.log
- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adv2-suite-runtime-vulkan.log
- ~/.cache/modelspec-wave-m1-a/m1-inspect-device/scratch/adv2-lint.log
- harness task output files under ~/.cache/claude-tmp/claude-1000/…/tasks/ (b1j92i5xs.output, bwb863dov.output)

Inside the worktree's target/: target/tmp/adversary2_inspect_pre_1_2/ and target/tmp/adversary2_create_device_old/ (the C source and libfake_vulkan.so).

## 7. Findings block

```findings
[
 {
  "file": "crates/modelspec-runtime/src/vulkan/probe.rs",
  "line": 239,
  "category": "contract-drift",
  "severity": "warning",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "below Vulkan 1.2 the probe leaves subgroup_size 0 and subgroup stages/operations empty though they are core in 1.1, so the printed DeviceProfile breaks the spec invariant subgroup_size > 0 and fails from_json; red in adversary2_inspect_pre_1_2.rs, reached only through a constructed loader"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/probe.rs",
  "line": 255,
  "category": "boundary",
  "severity": "warning",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "spirv_version reports SPIR-V 1.5 for Vulkan 1.0 and 1.1 devices, which accept at most SPIR-V 1.0 and 1.3, a false capability claim; red in adversary2_inspect_pre_1_2.rs"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/mod.rs",
  "line": 100,
  "category": "acceptance",
  "severity": "note",
  "verdict": "INFEASIBLE",
  "origin": "introduced",
  "message": "create_device does not refuse a device below 1.2 and reports api_version 1.2 for a logical device made on a 1.1 device when given another device's Negotiated; red in adversary2_create_device_old.rs, no caller builds that input"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/probe.rs",
  "line": 237,
  "category": "contract-drift",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "below 1.2 device_uuid is null and f16_storage and the three float-control fields are false, recorded as driver values the driver never stated, though 1.1 exposes the UUID and 16-bit storage"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/enabled.rs",
  "line": 128,
  "category": "judgement",
  "severity": "note",
  "verdict": "INFEASIBLE",
  "origin": "introduced",
  "message": "enabled::of ignores the pNext chain, so features enabled through VkPhysicalDeviceFeatures2 or the 1.1/1.2 feature structs would read back as none; create_device chains nothing today"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/negotiate.rs",
  "line": 63,
  "category": "judgement",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "the E-TARGET-004 diagnostic names no device index, and the json listing carries no verdict, so with several refused devices the stderr lines cannot be told apart"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/probe.rs",
  "line": 106,
  "category": "boundary",
  "severity": "note",
  "verdict": "INFEASIBLE",
  "origin": "introduced",
  "message": "spirv_version is given the unmasked api_version, so a non-zero variant makes any device compare as 1.3 or later and claim SPIR-V 1.6"
 }
]
```
