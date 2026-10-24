---
format: aep.planning-md/3
id: story:matmul-reads-transposed-operand
kind: story
status: draft
title: A matmul reads a transposed operand in place, so a step dispatches no transpose
relations:
- serves: vision:modelspec
- depends_on: story:matmul-shape-aware-tiling
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: cited
  path: website/docs/guides/inspect-a-run.md
revision: 2
---
# A matmul reads a transposed operand in place, so a step dispatches no transpose

## Context

Timo, 2026-10-24: "modelspec beats python+cuda in the bigger batch benchmark". After
story:matmul-shape-aware-tiling the `mlp_sweep` step at batch 8192, hidden 4096 is 13,285.0 µs
against 7,290.5 µs for PyTorch eager CUDA (ratio 1.82). `modelspec ir --stage kernel` on that
source shows three transpose kernels per step: k7 transposes the activation `[8192, 4096]` to
`[4096, 8192]` for the second weight gradient, k10 transposes the input `x` `[8192, 784]` to
`[784, 8192]` for the first, and k5 transposes `w2` `[4096, 10]` for the hidden gradient. The
profiled step of `benchmarks/raw/2026-10-24-tiling/after-profile-b8192-h4096.txt` spends
2,000.5 µs of 15,731 µs in k7 alone (268,435,456 bytes moved, 0 FLOP), and k10 moves
51,380,224 bytes every step although `x` does not change between steps. The transpose kernels
exist only because `Shell::matmul` stages both operands row-major from a contiguous view.

story:weight-grad-matmul-structural names "reading the transposed operand strided (dropping the
transpose kernel)" as one of three candidates for the MNIST weight gradient (k11, 4.7 µs); this
story takes that candidate for every shape and leaves the other two candidates there.

## Outcome

A matmul whose operand is the transpose of a contiguous tensor reads that tensor in place:
the kernel plan carries, per operand, whether it is read transposed, and the kernel stages the
operand's tile from the untransposed view with the strides swapped. The schedule emits no
`Transpose` dispatch for an operand that only a matmul reads, so the step above has no k5, k7
or k10 and the two weight-gradient matmuls read `x` and the activation directly. Every output
element keeps its order: one invocation, ascending `k` from the `k = 0` product, no split-K,
`NoContraction` on every multiply and add, so the bits are those of the plan before the change.

## Acceptance

- **transposed-operand-flag**: `MatMulKernel` (or the kernel record that `ir --stage kernel`
  prints) states per operand whether it is read transposed; the printed line shows it, the
  kernel signature and module id change with it, and `MatMulTile::of` still takes the logical
  `(m, k, n)`.
- **no-transpose-dispatch**: on `examples/mnist_mlp.mspec` and on the sweep source at b8192
  h4096, `ir --stage kernel` lists no `transpose` kernel whose output only a matmul reads; a
  transpose whose output another kernel reads stays. The MNIST golden
  `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir` and the kernel lines of
  `website/docs/guides/inspect-a-run.md` are regenerated in the same change and
  `crates/modelspec-cli/tests/ir_kernel.rs` and `adversary_guide_kernel_lines.rs` pass.
- **same-bits**: for every matmul site of the MNIST step plan and of the sweep plans at b1000
  h1024 and b8192 h1024 and h4096, the device output is bit-identical to the plan before the
  change on the same device (a device test against recorded digests, as
  `crates/modelspec-runtime/tests/vulkan_matmul_same_bits.rs` does), and `train --backend
  vulkan` on those sources prints the same parameters as before (sha256 of stdout equal).
- **staged-reads-coalesced**: the transposed operand's tile is staged so that consecutive
  invocations read consecutive addresses of the untransposed view; the story reports the
  per-matmul `mean_us` and GFLOP/s of k8 and k11 at b8192 h4096 from `--profile` before and
  after, and the MNIST k11 figure beside the 4.716 µs of story:weight-grad-matmul-structural.
- **sweep-median**: the `mlp_sweep` median at b8192 h4096 after the change is reported beside
  13,285.0 µs (before) and 7,290.5 µs (PyTorch eager CUDA), with the ratio; medians of 3 timed
  runs of at least 5 s, GPU to itself, raw records under `benchmarks/raw/<date>-transposed/`.
- **design-states-rule**: DESIGN.md § 3.1 and § 5.2 state when a transpose is folded into its
  matmul reader and that the output order is unchanged; the revision line is bumped.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-runtime/src/vulkan/dispatch.rs`
- `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`
- `website/docs/guides/inspect-a-run.md`
- `DESIGN.md`
