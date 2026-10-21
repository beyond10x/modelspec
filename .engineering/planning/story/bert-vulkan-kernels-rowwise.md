---
format: aep.planning-md/3
id: story:bert-vulkan-kernels-rowwise
kind: story
status: draft
title: Vulkan kernels for the row-wise BERT operators, checked against the CPU oracle
summary: SPIR-V kernels for embedding, add_rows, gelu, dropout, layer_norm and softmax with their gradients, within 1e-5 absolute plus 1e-4 relative of the CPU oracle on a device, tolerances recorded in DESIGN.md first; device runs are operator-started.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-tiny-cpu-model
- depends_on: epic:m1-vulkan-execution
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-docs/src/status.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/
- confidence: cited
  path: crates/modelspec-runtime/tests/spirv_bert_rowwise.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/vulkan_bert_rowwise.rs
- confidence: cited
  path: website/data/status.json
revision: 4
---
# Vulkan kernels for the row-wise BERT operators, checked against the CPU oracle

## Context

`epic:bert-from-scratch` (In scope, Execution: "the GPU backend (epic:m1-vulkan-execution) for the
full run; the CPU interpreter for gradient checks on a 2-layer, hidden-32 configuration"). The epic
depends on `epic:m1-vulkan-execution`, which builds the `ash` runtime, `rspirv` emission and the M1
kernels and has no stories yet. DESIGN.md § 9 orders the gates: CPU oracle and finite-difference
checks (gates 1-3) before GPU work, then Vulkan against CPU (gate 5) and `spirv-val` on every
generated module (gate 6). The CPU side of every operator here is done by the operator stories,
from `story:bert-embedding-lookup` to `story:bert-adamw-warmup`.

## What this story decides

- SPIR-V kernels, behind the `vulkan` feature of `modelspec-runtime` (DESIGN.md § 7), for
  `Embedding`, `EmbeddingGrad`, `AddRows`, `SumRows`, `Gelu`, `GeluGrad`, `Dropout`, `DropoutGrad`,
  `LayerNorm`, `LayerNormGradX`, `LayerNormGradGamma`, `Softmax` and `SoftmaxGrad`, with `i32`
  buffers for ids and tokens.
- `EmbeddingGrad` accumulates without floating-point atomics, each output row summing its
  contributions in ascending `n`, so a deterministic compile gives the same bits on every run
  (`deterministic` of `modelspec.artifact.CompiledArtifact`: "no floating-point reduction is
  reassociated").
- `Dropout` computes Philox4x32-10 in the kernel from the token of `story:bert-dropout`, and its
  mask equals the CPU mask bit for bit.

## Tolerances, settled first

DESIGN.md § 9 says "GPU results use documented absolute/relative tolerances, not bit-for-bit
equality", and neither DESIGN.md nor `epic:m1-vulkan-execution` holds a number. This story fixes
the numbers before any device scenario runs:

- A kernel passes when every element `g` of its GPU result and the element `c` of the CPU
  interpreter result, both `f32`, satisfy `|g - c| <= 1e-5 + 1e-4 * |c|`: absolute tolerance
  `GPU_ATOL = 1e-5`, relative tolerance `GPU_RTOL = 1e-4`. The relative part is about 840 `f32`
  units in the last place, above the worst-case error `512 * 2^-23 = 6.1e-5` of a 512-term sum
  added in another order, so device tests keep every reduction at most 512 long.
- Dropout masks equal the CPU masks bit for bit, and `EmbeddingGrad` gives identical bits in two
  GPU runs.
- First step of the story, before any kernel code: the two constants and the comparison rule are
  written into DESIGN.md § 9 (gate 5) with a revision-line entry, and the device tests read
  `GPU_ATOL` and `GPU_RTOL` from one place. If `epic:m1-vulkan-execution` has documented other
  gate-5 numbers by the time this story starts, this body is revised to them before it starts;
  it is not run against either set silently.

## Order on shared files

This story edits the Vulkan and SPIR-V modules of `crates/modelspec-runtime/` (DESIGN.md § 7:
`vulkan/`, `spirv/`), which `epic:m1-vulkan-execution` creates, and `DESIGN.md` § 9 and its
revision line, `CHANGELOG.md`, `crates/modelspec-docs/src/status.rs` and `website/data/status.json`
after `story:bert-tiny-cpu-model`. It depends on both.
`story:bert-vulkan-kernels-attention-loss-adamw` follows it.

## Domain relations

- Each kernel is a `modelspec.artifact.KernelModule` owned by its `CompiledArtifact` (relation
  `kernels`, `kind: owns`, `spec/domains/artifact.yaml`); this story adds kernels of that type and
  no relation.

## Scope

- `crates/modelspec-runtime/src/spirv/` and `src/vulkan/` (the files `epic:m1-vulkan-execution`
  creates there): emission and dispatch of the thirteen kernels.
- `crates/modelspec-runtime/tests/spirv_bert_rowwise.rs` and
  `crates/modelspec-runtime/tests/vulkan_bert_rowwise.rs` (new), beside the M1 device tests;
  `GPU_ATOL` and `GPU_RTOL` in a module those tests share.
- `DESIGN.md` § 9 (gate 5 tolerances) and the revision line.
- `crates/modelspec-docs/src/status.rs` (`CAPABILITIES`) and `website/data/status.json`;
  `CHANGELOG.md` one line.

## Acceptance

The thirteen row-wise kernels pass `spirv-val`, and on a Vulkan device every element of their
forward and gradient results is within `1e-5 + 1e-4 * |c|` of the CPU interpreter value `c`, with
dropout masks and the embedding gradient bit-equal.

Scenarios:

1. `grep -Eq "^\*\*Revision:\*\* [0-9.]+ \([0-9-]+\): [^;]*tolerance" DESIGN.md` exits 0, and
   DESIGN.md § 9 states `1e-5` absolute and `1e-4` relative for gate 5; run by hand before any
   device scenario.
2. `cargo test -p modelspec-runtime --features vulkan --test spirv_bert_rowwise`: every module
   passes `spirv-val`. It needs no device; it is part of `task ci` if `epic:m1-vulkan-execution`
   puts the `vulkan` feature and `spirv-val` into `task ci` as gate 6 asks, which I do not know yet.
3. Device run, started by the operator on a machine with a Vulkan device, not part of `task ci`:
   `cargo test -p modelspec-runtime --features vulkan --test vulkan_bert_rowwise -- --ignored`:
   for each operator, on shapes that include edges not aligned to the workgroup size and
   reductions of at most 512 terms, every forward and gradient element satisfies
   `|g - c| <= 1e-5 + 1e-4 * |c|` against the CPU interpreter.
4. Device run, as scenario 3: `Dropout` masks equal the CPU masks bit for bit for three seeds and
   two steps, and `EmbeddingGrad` with repeated ids gives identical bits in two runs.
5. `task ci` exits 0 without a Vulkan device.

## Out of scope

- Attention, cross-entropy and AdamW kernels (`story:bert-vulkan-kernels-attention-loss-adamw`).
- Fused or tuned kernels (DESIGN.md § 5.2 after M1).
