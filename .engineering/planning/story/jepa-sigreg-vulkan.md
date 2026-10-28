---
format: aep.planning-md/3
id: story:jepa-sigreg-vulkan
kind: story
status: proposed
title: sigreg on Vulkan, held to the CPU oracle
summary: Two GPU kernels for SIGReg with FLOP counting rules and per-step projection upload, within the run tolerance of the CPU.
relations:
- decomposes: epic:jepa-operators
- depends_on: story:jepa-sigreg-cpu
- serves: vision:modelspec
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests
- confidence: cited
  path: crates/modelspec-core/src
- confidence: cited
  path: crates/modelspec-docs/src
- confidence: cited
  path: crates/modelspec-runtime/src/spirv
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan
- confidence: cited
  path: crates/modelspec-runtime/tests
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T10:45:53Z", actor: "human:timo", revision: 2}
---

`id: story:jepa-sigreg-vulkan` · depends_on: `story:jepa-sigreg-cpu` · size: about one day

## Outcome

`train --backend vulkan` runs a program with `sigreg`. `SigregColumns` and `SigregColumnsGrad` are
per-column kernels that run the CPU's operations in its order, with the same `sin`/`cos` routine.
The projection matrix of each site is uploaded before each step, as dropout tokens are. Each kernel
has a FLOP counting rule.

## Context

- Kernel kinds: `PerColumn` (`crates/modelspec-core/src/ir/kernel.rs:159`), used by `SumRows` and
  `LayerNormGradGamma`. Op to kernel: `schedule::rowwise_kernel` (`schedule.rs:1211`, per-column
  arms at `:1155`, `:1180`). Emission: `spirv/emit/rowwise.rs:179` and `:492` (`SumRows`),
  dispatched from `spirv/emit.rs:398`. `GLSL.std.450` access: `emit.rs:1319`.
- FLOP rules: `Kernel::flop` (`kernel.rs:984`, `PerColumn` arm at its end).
- Per-step host inputs: `vulkan/stepwise.rs:125` (`needed`) and `:499` (token upload).
  `vulkan/runner.rs:247-258` sends dropout and AdamW programs stepwise.
- Comparison classes and tolerances: DESIGN.md § 5.4 (lines 345-349) and § 9 gate 5 (line 598).
  `story:bert-vulkan-kernels-rowwise` is the precedent: commit `7770108` touched 26 files, among
  them `tests/spirv_bert_rowwise.rs` and `tests/vulkan_bert_rowwise.rs`.

## Scope

- `crates/modelspec-core/src/schedule.rs`, `ir/kernel.rs` (kinds doc and `flop`: proposed
  `M·(8·(5N + 8) + 1)` for `SigregColumns` and `M·8·(11N + 6)` for `SigregColumnsGrad`; the story
  fixes them in DESIGN.md § 5.2), `artifact.rs` if the plan names the op.
- `crates/modelspec-runtime/src/spirv/emit/rowwise.rs` (the two kernels and the `sin`/`cos`
  sequence, every `OpFAdd`/`OpFSub`/`OpFMul`/`Fma` `NoContraction` outside `fast_math`),
  `spirv/emit.rs`, `vulkan/stepwise.rs` (`needed` true for a `sigreg` site; per-step projection
  upload), `vulkan/runner.rs`, `vulkan/pipeline.rs`.
- Tests: `crates/modelspec-runtime/tests/spirv_sigreg.rs` (new; `spirv-val` of both modules),
  `crates/modelspec-runtime/tests/vulkan_sigreg.rs` (new, `#[ignore]`d device test),
  `crates/modelspec-cli/tests/train_vulkan.rs` (one `sigreg` fixture case).
- DESIGN.md § 5.2 (kernel bullet, counting rows), § 5.4 (class: within tolerance) and the revision
  line. `crates/modelspec-docs/src/status.rs`, then `task docs-generate`. One line in
  `CHANGELOG.md`.

## Acceptance

`MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_sigreg -- --ignored`
passes on the RTX 3090:

- Shapes: `N ∈ {1, 63, 64, 65, 256}`, `M ∈ {1, 63, 64, 65, 1024}`, `K = 8`.
- Every element of `SigregColumns` and `SigregColumnsGrad` is within
  `GPU_ATOL + GPU_RTOL · r` of the CPU interpreter. `r` is the sum of the absolute values of the
  column's terms (the reduction rule of gate 5).
- The 16 guard words after each output are unchanged, and no validation message appears.

## Done when

- `cargo test -p modelspec-runtime --features vulkan --test spirv_sigreg` passes `spirv-val`
  (in `task test`).
- `cargo test -p modelspec-cli --test train_vulkan -- --ignored`: the S2 `sigreg` fixture trained
  on Vulkan matches the CPU run within `GPU_RUN_ATOL`/`GPU_RUN_RTOL` at every step loss and every
  parameter element.
- `task ci` is green. The test records whether the two kernels were bit-equal to the CPU on this
  device; that is not required.

## Out of scope

- Speed: one dispatch per op, no fusion of the projection matmul into the column kernel.
- `ema_update` and `fast_math` variants.
