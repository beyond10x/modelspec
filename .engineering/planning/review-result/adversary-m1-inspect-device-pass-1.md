---
format: aep.planning-md/3
id: review-result:adversary-m1-inspect-device-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-inspect-device
relations:
- reviews: story:m1-inspect-device
revision: 1
---
# Adversary pass 1 on story:m1-inspect-device

Verdict: CONFIRMED contract drift; 3 cases added in
`crates/modelspec-cli/tests/adversary_inspect_old_device.rs` (stub loader: device 0 at 1.2.0,
device 1 at 1.1.0), all red. `cargo test --workspace --no-fail-fast`: 659 passed, 3 failed, 1
ignored. `core-no-vulkan` (7), `cli-without-vulkan` (4), `check-examples`, `docs-check`,
`spec-validate` exit 0; the ignored device test passes on the RTX 3090. Sent back to the
implementor with findings 1, 2 and 4 to fix.

Attacked without a break: no Vulkan crate in `modelspec-core` (`cargo tree --all-features --target
all`), E-TARGET-001 without the feature, DeviceProfile against `spec/domains/device.yaml` field by
field, E-TARGET-002 for a missing path and a non-Vulkan `.so`, E-TARGET-003 on the real device,
the 24 text fields plus `negotiation: ok`, the fixture round trip (live runs differ only in
`device_profile_id` and heap 0 `budget_bytes`), instance and extension-name lifetimes.

```findings
[
 {
  "file": "crates/modelspec-runtime/src/vulkan/probe.rs",
  "line": 127,
  "category": "contract-drift",
  "severity": "warning",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "the probe refuses a device below Vulkan 1.2 for the whole enumeration, so --device 0 on a valid device, --device 99 (E-TARGET-003) and the documented \"negotiation: refused (E-TARGET-004)\" listing all end in a bare E-TARGET-004; red in adversary_inspect_old_device.rs, reached only by a constructed loader"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/mod.rs",
  "line": 148,
  "category": "judgement",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "Device::enabled_features and api_version are constants, so the device test's \"created with the negotiated set\" assertion cannot fail for features or version"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/mod.rs",
  "line": 46,
  "category": "boundary",
  "severity": "note",
  "verdict": "INFEASIBLE",
  "origin": "introduced",
  "message": "MODELSPEC_VULKAN_LIBRARY pointing at a Vulkan layer .so segfaults (exit 139) instead of failing with a stated code; non-Vulkan .so files correctly give E-TARGET-002"
 },
 {
  "file": "crates/modelspec-runtime/src/vulkan/probe.rs",
  "line": 190,
  "category": "judgement",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "introduced",
  "message": "an NV-only cooperative-matrix driver gets cooperative_matrix true with no shapes, and name_of uses an unbounded CStr::from_ptr where ash offers a bounded accessor"
 },
 {
  "file": "DESIGN.md",
  "line": 581,
  "category": "judgement",
  "severity": "note",
  "verdict": "CONFIRMED",
  "origin": "pre-existing",
  "message": "the RTX 3090 reports shaderDenormPreserveFloat32 false, so under § 5.4 nothing on the target device is bit-for-bit and ADR-007's bit-for-bit clause is empty there"
 }
]
```
