---
format: aep.planning-md/3
id: review-result:m1-g-compile-adversary-1
kind: review-result
status: active
title: 'm1-g unit 3 adversary pass 1: 6 red (stale kernels/, unbounded module reads, buffer limit, SPIR-V), 1 note'
relations:
- reviews: story:m1-compile-artifact
revision: 1
---
```
unit: story:m1-compile-artifact, branch impl/m1-compile-artifact @ 7ea86f9 (base c58fcc9), plus 2 untracked test files
verdict: NEEDS-CHANGE
cases: executed 18→24, red 6
origin: introduced 6 / pre-existing 0 / undecided 1
```

1. **Cases added**, all red on 7ea86f9:

| Test | Red output |
|---|---|
| `a_cpu_compile_into_a_reused_vulkan_output_leaves_no_kernels_dir` | the Cpu artifact still has kernels/ with 14 stale .spv files |
| `a_vulkan_compile_into_a_reused_output_has_one_entry_per_kernel_file` | kernels/ holds stale modules of the earlier artifact, left: 21 right: 7 |
| `a_module_symlinked_to_dev_zero_is_refused_with_004` | E-ARTIFACT-002 instead of 004 |
| `a_module_that_is_a_fifo_is_refused_without_blocking` | artifact::read blocked for 10 s on a FIFO module file |
| `a_buffer_past_u32_max_re_sealed_is_refused_with_003` | accepted |
| `a_module_that_is_not_spirv_is_refused` (core) | 14 kernel modules of 3 bytes each accepted |

2. **Suites**: `compile` 9 passed; `compile_adversary` 9 passed; `adversary_compile_artifact` 0 passed, 5 failed; `adversary_artifact_reader` 0 passed, 1 failed. Clippy and fmt clean.

3. **Implementor's deviations**: blanking `artifact_id` with the hash is correct (the id derives from it); the byte-size check before the verifier is harmless (same code 003); `canonical_parameter_id` as a JSON u64 is exact in serde_json; f32-bit immediates read back exactly.

4. **Attacked, not broken**: check order 001/004/002/003; path traversal (ids are canonical UUIDs); duplicate entries (003); unknown JSON fields (001); a 64 MiB manifest cap; a shorter module (002); an unowned Dispatch module (003); shape overflow (003); Cpu artifact without kernels/; E-TARGET-004 before any write; `--output` on a file; determinism; no panic on any altered input. Not reported as a finding: the hash covers the re-serialized manifest, so reformatted JSON or duplicate keys still pass 002.

```findings
- file: crates/modelspec-core/src/artifact.rs
  line: 283
  category: acceptance
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: compile into a reused --output leaves stale kernels/*.spv (14 after a Cpu compile, 21 files for 7 entries after another Vulkan compile), breaking acceptance 1 and 3 while the reader still accepts the directory
- file: crates/modelspec-core/src/artifact.rs
  line: 360
  category: boundary
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: a module longer than its entry but reported as length 0 by metadata (symlink to /dev/zero) is read and hashed and refused with 002 instead of 004, and the read is bounded only by the untrusted byte_length
- file: crates/modelspec-core/src/artifact.rs
  line: 348
  category: boundary
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: artifact::read blocks indefinitely when a module file is a FIFO; non-regular module files are not refused before open
- file: crates/modelspec-core/src/artifact.rs
  line: 1181
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the reader accepts a re-sealed buffer of 2^33 bytes although artifact.yaml says no buffer exceeds 4294967295 bytes
- file: crates/modelspec-core/src/artifact.rs
  line: 359
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: neither write nor read checks that module bytes are SPIR-V (magic number, whole words), so 3-byte text modules are written and accepted
- file: crates/modelspec-core/src/ir/verify.rs
  line: 1214
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: undecided
  message: the plan verifier's per-step closure sets grow with the square of the step count, and artifact::read now feeds it untrusted plans of up to 64 MiB, so a re-sealed long dependency chain exhausts memory
```
