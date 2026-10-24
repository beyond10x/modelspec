---
format: aep.planning-md/3
id: story:matmul-reads-transposed-operand
kind: story
status: implemented
title: A matmul reads a transposed operand in place, so a step dispatches no transpose
relations:
- serves: vision:modelspec
- depends_on: story:matmul-shape-aware-tiling
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/after2-fastmath-profile-b8192-h4096.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/after2-fastmath-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/after2-fastmath-sweep.stderr.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/after2-profile-b8192-h4096.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/after2-profile-mnist.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/after2-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/after2-sweep.stderr.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-fastmath-profile-b8192-h4096.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-fastmath-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-fastmath-sweep.stderr.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-profile-b8192-h4096.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-profile-mnist.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-r2-fastmath-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-r2-fastmath-sweep.stderr.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-r2-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-r2-sweep.stderr.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/baseline-sweep.stderr.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/same-bits-train-stdout-sha256.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/superseded-kmajor-fastmath-profile-b8192-h4096.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/superseded-kmajor-fastmath-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/superseded-kmajor-fastmath-sweep.stderr.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/superseded-kmajor-profile-b8192-h4096.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/superseded-kmajor-profile-mnist.txt
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/superseded-kmajor-sweep.jsonl
- confidence: cited
  path: benchmarks/raw/2026-10-24-transposed/superseded-kmajor-sweep.stderr.txt
- confidence: cited
  path: crates/modelspec-cli/tests/adversary_step_counters.rs
- confidence: cited
  path: crates/modelspec-cli/tests/compile.rs
- confidence: cited
  path: crates/modelspec-cli/tests/golden/linear_regression.kernel.ir
- confidence: cited
  path: crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir
- confidence: cited
  path: crates/modelspec-cli/tests/ir_kernel.rs
- confidence: cited
  path: crates/modelspec-cli/tests/matmul_transposed_artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-core/tests/adversary_dispatch_count.rs
- confidence: cited
  path: crates/modelspec-core/tests/adversary_fuse.rs
- confidence: cited
  path: crates/modelspec-core/tests/kernel_plan.rs
- confidence: cited
  path: crates/modelspec-core/tests/matmul_transposed_flag.rs
- confidence: cited
  path: crates/modelspec-core/tests/matmul_transposed_operand.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/pipeline.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_fastmath_emission.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_loss_sum.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/adversary_step_counters.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/common/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/fixtures/matmul_plan_sites_same_bits.txt
- confidence: cited
  path: crates/modelspec-runtime/tests/fixtures/untransposed_modules.txt
- confidence: cited
  path: crates/modelspec-runtime/tests/matmul_untransposed_modules_unchanged.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/spirv_val.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/vulkan_matmul_plan_sites_same_bits.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/vulkan_matmul_transposed.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/vulkan_profile.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/vulkan_train.rs
- confidence: cited
  path: spec/domains/artifact.yaml
- confidence: cited
  path: website/docs/guides/inspect-a-run.md
revision: 8
transitions:
- {from: "draft", to: "proposed", at: "2026-10-24T13:20:29Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-10-24T13:20:29Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-10-24T15:49:26Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"test_result":1,"review_outcome":2,"verification":1}}}
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

- `benchmarks/raw/2026-10-24-transposed/after2-fastmath-profile-b8192-h4096.txt`
- `benchmarks/raw/2026-10-24-transposed/after2-fastmath-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/after2-fastmath-sweep.stderr.txt`
- `benchmarks/raw/2026-10-24-transposed/after2-profile-b8192-h4096.txt`
- `benchmarks/raw/2026-10-24-transposed/after2-profile-mnist.txt`
- `benchmarks/raw/2026-10-24-transposed/after2-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/after2-sweep.stderr.txt`
- `benchmarks/raw/2026-10-24-transposed/baseline-fastmath-profile-b8192-h4096.txt`
- `benchmarks/raw/2026-10-24-transposed/baseline-fastmath-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/baseline-fastmath-sweep.stderr.txt`
- `benchmarks/raw/2026-10-24-transposed/baseline-profile-b8192-h4096.txt`
- `benchmarks/raw/2026-10-24-transposed/baseline-profile-mnist.txt`
- `benchmarks/raw/2026-10-24-transposed/baseline-r2-fastmath-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/baseline-r2-fastmath-sweep.stderr.txt`
- `benchmarks/raw/2026-10-24-transposed/baseline-r2-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/baseline-r2-sweep.stderr.txt`
- `benchmarks/raw/2026-10-24-transposed/baseline-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/baseline-sweep.stderr.txt`
- `benchmarks/raw/2026-10-24-transposed/same-bits-train-stdout-sha256.txt`
- `benchmarks/raw/2026-10-24-transposed/superseded-kmajor-fastmath-profile-b8192-h4096.txt`
- `benchmarks/raw/2026-10-24-transposed/superseded-kmajor-fastmath-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/superseded-kmajor-fastmath-sweep.stderr.txt`
- `benchmarks/raw/2026-10-24-transposed/superseded-kmajor-profile-b8192-h4096.txt`
- `benchmarks/raw/2026-10-24-transposed/superseded-kmajor-profile-mnist.txt`
- `benchmarks/raw/2026-10-24-transposed/superseded-kmajor-sweep.jsonl`
- `benchmarks/raw/2026-10-24-transposed/superseded-kmajor-sweep.stderr.txt`
- `crates/modelspec-cli/tests/adversary_step_counters.rs`
- `crates/modelspec-cli/tests/compile.rs`
- `crates/modelspec-cli/tests/golden/linear_regression.kernel.ir`
- `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`
- `crates/modelspec-cli/tests/ir_kernel.rs`
- `crates/modelspec-cli/tests/matmul_transposed_artifact.rs`
- `crates/modelspec-core/src/artifact.rs`
- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-core/src/ir/print.rs`
- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/tests/adversary_dispatch_count.rs`
- `crates/modelspec-core/tests/adversary_fuse.rs`
- `crates/modelspec-core/tests/kernel_plan.rs`
- `crates/modelspec-core/tests/matmul_transposed_flag.rs`
- `crates/modelspec-core/tests/matmul_transposed_operand.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `crates/modelspec-runtime/src/vulkan/pipeline.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`
- `crates/modelspec-runtime/tests/adversary_fastmath_emission.rs`
- `crates/modelspec-runtime/tests/adversary_loss_sum.rs`
- `crates/modelspec-runtime/tests/adversary_step_counters.rs`
- `crates/modelspec-runtime/tests/common/mod.rs`
- `crates/modelspec-runtime/tests/fixtures/matmul_plan_sites_same_bits.txt`
- `crates/modelspec-runtime/tests/fixtures/untransposed_modules.txt`
- `crates/modelspec-runtime/tests/matmul_untransposed_modules_unchanged.rs`
- `crates/modelspec-runtime/tests/spirv_val.rs`
- `crates/modelspec-runtime/tests/vulkan_matmul_plan_sites_same_bits.rs`
- `crates/modelspec-runtime/tests/vulkan_matmul_transposed.rs`
- `crates/modelspec-runtime/tests/vulkan_profile.rs`
- `crates/modelspec-runtime/tests/vulkan_train.rs`
- `DESIGN.md`
- `spec/domains/artifact.yaml`
- `website/docs/guides/inspect-a-run.md`



## Closing note (2026-10-24, coordinator)

The `test_result` recorded at 15:49:26Z names "8d58780 task ci exit 0"; that run exited 201 at
`schema-check` (the spec summary change in `spec/domains/artifact.yaml` changed the projection's
source digest and every schema file needed regeneration). The record is wrong and stays in the
history. The regeneration is the next commit on the branch; `task ci` steps 1 to `docs-check`
passed in that run (237 suites ok) and `schema-check`, `ui-types-check` and `ui-browser` passed
on the regenerated tree; the `test_result` recorded after this note names that run.
