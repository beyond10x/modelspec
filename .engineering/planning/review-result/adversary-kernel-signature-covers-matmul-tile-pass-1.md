---
format: aep.planning-md/3
id: review-result:adversary-kernel-signature-covers-matmul-tile-pass-1
kind: review-result
status: active
title: 'Adversary pass 1: the guide''s profile block mixed new ids with old times'
relations:
- reviews: story:kernel-signature-covers-matmul-tile
revision: 1
---
# Adversary pass 1 on story:kernel-signature-covers-matmul-tile

Tree at 336b901. Verdict: one warning, two notes, no failing case.

Ran: `cargo test --workspace` 970 passed, 0 failed, 60 ignored; `cargo test -p modelspec-runtime
--features vulkan` 324 passed, 0 failed, 50 ignored; the 1,800-step Vulkan run's stdout sha256
`b9bb028a…6904`; 4 profile runs whose ids match the `ir --stage kernel` golden.

Attacked without a break: Display ambiguity (every field labelled), signature tile against emitted
tile (both `MatMulTile::of`), transposed, fused-epilogue and weight-grad matmuls, SPIR-V caching by
signature (none), kernels that should share an id, 0.5.0 artifacts, Display and shape mutants.

Fixed in c89ea1b: the guide's profile block is one run at 336b901; `print_kernel`'s doc names the
tile suffix. Left: the SPIR-V shape constants outside `MatMulTile` (pre-existing, outside the
story).

```findings
- file: website/docs/guides/inspect-a-run.md
  line: 119
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: "the hand-edited profile line names k11's new depth-16 module id 73027c05 beside 4.831 us measured at ab375a5 on the depth-32 ahead-6 module, inside a block presented as one run"
- file: crates/modelspec-core/src/ir/print.rs
  line: 121
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "print_kernel's doc comment says a kernel line ends at workgroups [x, y, z] with the kind's workgroup size, but matmul lines now end with the tile suffix"
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: 636
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: "MATMUL_UNROLL and the b_stride padding shape matmul SPIR-V outside MatMulTile, so changing them keeps the module id (from reading the code, not run)"
```
