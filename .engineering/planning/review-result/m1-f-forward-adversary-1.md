---
format: aep.planning-md/3
id: review-result:m1-f-forward-adversary-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-f-forward-matmul-tiled
relations:
- reviews: story:m1-f-forward-matmul-tiled
revision: 1
---
# Adversary pass 1 on story:m1-f-forward-matmul-tiled

- Unit commit `a7d7903` on `impl/m1-f-forward-matmul-tiled`, with two adversary test files left
  untracked in the unit's worktree: `crates/modelspec-runtime/tests/adversary_matmul_tiled_spirv.rs`
  and `crates/modelspec-runtime/tests/adversary_vulkan_matmul_tiled.rs`.
- Verdict NEEDS-CHANGE: 2 blockers, both introduced by the fully unrolled tile loop.
- Bit identity held on the RTX 3090: all 13 `MATMUL_SHAPES` × 3 epilogues × signed and unsigned
  inputs were bit-equal to the CPU, and so was `[524297,3]x[3,16]` on a `[65535, 2, 1]` grid.
- The suite and `task ci` were not run: the coordinator held builds for disk space (19G free on `/`).
  The k = 600,000 pipeline probe was stopped by the coordinator by PID before it reached its assert.

## Red runs

`every_matmul_module_of_a_full_batch_weight_gradient_passes_spirv_val` (in `task ci`), exit 101:

```
[4, 600000] x [600000, 1] ... workgroups [1, 1, 1]: 48150927 words, id bound 8625176
spirv-val --target-env vulkan1.2 refused the module: error: line 0: Invalid SPIR-V.  The id bound is larger than the max id bound 4194303.
```

`a_large_k_matmul_module_is_emitted_and_its_pipeline_created_within_a_second` (device, ignored):

```
[100, 784] x [784, 64]: 40778 words, id bound 7424, emit 0.005 s, pipeline 0.001 s, created
[64, 4096] x [4096, 64]: 204905 words, id bound 37193, emit 0.025 s, pipeline 1.321 s, created
[784, 60000] x [60000, 64]: 2983835 words, id bound 541259, emit 0.362 s, pipeline 188.680 s, created
```

Logs: `~/.cache/modelspec-wave-m1-f/m1-f-forward-matmul-tiled/adversary/`.

```findings
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: 1538
  category: boundary
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the unrolled tile loop exceeds the SPIR-V id bound of 4,194,303 near k=290,000, so a full-batch weight-gradient matmul that the schedule accepts emits a module spirv-val refuses (acceptance 4)
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: 1538
  category: performance
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: pipeline creation of one matmul module takes 1.3 s at k=4096 and 188.7 s at k=60000 on the RTX 3090, against milliseconds for the base's looped kernel
- file: crates/modelspec-core/src/ir/kernel.rs
  line: 186
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: MatMulTile::of always sets rows to 1, so the register-blocking path is dead and untested although DESIGN.md and the commit message claim register blocking
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: 474
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: the SharedMemory refusal fires only for a profile reporting less than the Vulkan minimum of 16384 bytes, which no probed device can report
- file: crates/modelspec-runtime/tests/vulkan_matmul_tiled.rs
  line: 205
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: undecided
  message: on the tolerance-class RTX 3090 the unit's device test never asserts bit identity, so the claimed same bits are checked only through the stdout sha256 of training runs
```
