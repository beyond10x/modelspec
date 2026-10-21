---
format: aep.planning-md/3
id: story:m1-kernel-plan
kind: story
status: draft
title: Kernel IR, a fixed safe schedule and the execution plan of a training step, in modelspec-core
summary: Typed buffer views and a forward plan in spec, one kernel per op, a verified step plan with no completion point inside it, and modelspec ir --stage kernel
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/ir.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/golden/linear_regression.kernel.ir
- confidence: inferred
  path: crates/modelspec-cli/tests/ir_kernel.rs
- confidence: cited
  path: crates/modelspec-core/Cargo.toml
- confidence: inferred
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/compile.rs
- confidence: inferred
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/mod.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/lib.rs
- confidence: inferred
  path: crates/modelspec-core/src/schedule.rs
- confidence: inferred
  path: crates/modelspec-core/tests/kernel_plan.rs
- confidence: inferred
  path: crates/modelspec-core/tests/kernel_plan_adversary.rs
- confidence: cited
  path: crates/modelspec-docs/src/cli.rs
- confidence: cited
  path: spec/domains/artifact.yaml
- confidence: cited
  path: website/docs/reference
revision: 3
---
# Kernel IR, a fixed safe schedule and the execution plan of a training step, in modelspec-core

## Context

DESIGN.md § 3 puts two levels between the Training IR M0 builds and SPIR-V: Kernel IR ("low-level
loops, index expressions, dispatch dimensions, ... buffer spaces, layouts, barriers, and
lifetimes") and the Execution DAG ("dispatch and copy commands, resource dependencies/barriers,
scratch allocations, and CPU-visible completion points"). DESIGN.md § 5.3 fixes M1 to "a fixed
safe schedule", and § 7.1 has the compiler emit `CompiledArtifact { manifest, kernels, plan }`.
`spec/domains/artifact.yaml` types the artifact, its `PlanStep` and `BufferView`; it has no
UNMAPPED marker, but it cannot yet say which buffer view holds a parameter, an input, the loss or
the prediction, so no runtime could bind them. Nothing in the code builds either level:
`crates/modelspec-core/src/compile.rs` stops at `CompiledTraining`.

The plan this story fixes is what the 10x lane of `epic:pytorch-10x-mnist` needs from M1: one step
body that a runtime can record many times into one command buffer. So the step plan carries no
CPU-visible completion point; the runtime places one at the end of a run and at each requested
checkpoint.

## What it adds

- `spec/domains/artifact.yaml`: `ParameterSpec.view` (the buffer view that holds the parameter;
  the step reads version n and the update writes n + 1 into the same view), `DataSpec.view`,
  `CompiledArtifact.loss_view`, `CompiledArtifact.prediction_view` and
  `CompiledArtifact.forward_plan` (the inference plan: the forward pass without loss, gradients or
  update), and in the summary of `plan` that the step plan has no `Complete` step.
- `crates/modelspec-core/src/ir/kernel.rs`: one kernel per Tensor IR op of the training step,
  with its kind (elementwise, broadcast, reduce, transpose, matmul, SGD update), shapes, dtype,
  bindings, workgroup size and dispatch dimensions; a printer with stable ids.
- `crates/modelspec-core/src/schedule.rs`: the fixed safe schedule. Ops in Training IR order, one
  dispatch each, a barrier wherever a dispatch reads what an earlier one wrote, one buffer per
  value (no reuse), and each SGD update in place, placed after every read of the version it
  replaces.
- `crates/modelspec-core/src/artifact.rs`: `CompiledArtifact` for `Backend::Cpu` and
  `Backend::Vulkan` with manifest, plan and forward plan; `source_hash`, `ir_hash` and
  `schema_hash` (blake3 over the ordered parameter paths, dtypes and shapes, independent of the
  backend and of the learning rate). Kernel module ids derive from the content hash of each kernel,
  so two compiles of one file give equal ids. A Cpu artifact has an empty plan and no kernels.
- `crates/modelspec-core/src/ir/verify.rs`: a plan verifier. Every dependency lies before its
  step, every view lies inside its buffer, each update depends on every reader of the version it
  overwrites, and workgroup sizes stay within the limits Vulkan guarantees on every device (128
  invocations, 65,535 workgroups per dimension).
- `modelspec ir <file> --stage kernel`, printing the kernels and both plans.

## Domain relations

- CompiledArtifact owns KernelModule (`kernels`, `spec/domains/artifact.yaml:213`); a Dispatch
  names its module by `PlanStep.kernel_module_id` (line 105). Inferable from ess/1.
- A parameter has one view across steps: read at version n, written at n + 1
  (`crates/modelspec-core/src/ir/training.rs:127`). Inferable from code.
- One data view per input and target name (`crates/modelspec-runtime/src/train/data.rs:36`).
  Inferable from code.
- `forward_plan` is what `crates/modelspec-core/src/compile.rs:37` calls "what the data loader
  runs". Inferable from code.

## Scope

- `spec/domains/artifact.yaml`.
- `crates/modelspec-core/src/ir/kernel.rs` (new), `crates/modelspec-core/src/ir/mod.rs`,
  `crates/modelspec-core/src/ir/verify.rs`, `crates/modelspec-core/src/ir/print.rs`.
- `crates/modelspec-core/src/schedule.rs` (new), `crates/modelspec-core/src/artifact.rs` (new),
  `crates/modelspec-core/src/compile.rs`, `crates/modelspec-core/src/lib.rs`,
  `crates/modelspec-core/Cargo.toml`.
- `crates/modelspec-core/tests/kernel_plan.rs` (new),
  `crates/modelspec-core/tests/kernel_plan_adversary.rs` (new).
- `crates/modelspec-cli/src/ir.rs`, `crates/modelspec-cli/tests/ir_kernel.rs` (new),
  `crates/modelspec-cli/tests/golden/linear_regression.kernel.ir` (new).
- `crates/modelspec-docs/src/cli.rs`, `website/docs/reference/` through `task docs-generate`.
- `DESIGN.md` § 3.1 and § 7.1 (the views, the forward plan, no completion point in the step
  plan), revision line bumped. `CHANGELOG.md`.

## Acceptance

The compiler turns every M0 training step into verified Kernel IR and a plan whose views a runtime
can bind, with no completion point inside the step, and `ir --stage kernel` shows both.

1. First: `spec/domains/artifact.yaml` types `ParameterSpec.view`, `DataSpec.view`,
   `CompiledArtifact.loss_view`, `CompiledArtifact.prediction_view` and
   `CompiledArtifact.forward_plan`, and `ess specify validate --path .`, run from inside `spec/`,
   prints `valid` with no new warning.
2. `cargo test -p modelspec-core --test kernel_plan`: for `examples/linear_regression.mspec` and
   `crates/modelspec-docs/models/two_layer_relu.mspec` the step plan has exactly one Dispatch per
   computing op of the training step (every op but Input, ParamRead and Scalar), no `Complete`
   step, and passes the verifier; the forward plan writes no parameter view and has no SGD
   dispatch.
3. `cargo test -p modelspec-core --test kernel_plan_adversary`: the verifier refuses each of six
   altered plans with its own error: a dependency on a later step, an update that does not depend
   on a reader of its parameter, a view past the end of its buffer, a workgroup of 129
   invocations, 65,536 workgroups in one dimension, and a Dispatch without a kernel module.
4. Two compiles of one file give byte-equal printed Kernel IR and equal kernel module ids;
   `schema_hash` is equal for Cpu and Vulkan artifacts and for two learning rates, and differs when
   one parameter shape changes.
5. `cargo test -p modelspec-cli --test ir_kernel`: `modelspec ir examples/linear_regression.mspec
   --stage kernel` exits 0 and equals `crates/modelspec-cli/tests/golden/linear_regression.kernel.ir`
   byte for byte; `--stage kernels` exits 2.
6. `task ci` and `task core-no-vulkan` exit 0; `modelspec-core` gains no Vulkan, SPIR-V or GPU
   dependency.

## Out of scope

- SPIR-V emission and the artifact on disk: `story:m1-vulkan-elementwise`,
  `story:m1-vulkan-reduce-matmul`, `story:m1-compile-artifact`.
- Fusion, tiling, liveness-based buffer reuse and a whole-run schedule (M2,
  `epic:pytorch-10x-mnist`).
- Device ownership of a buffer view (`BufferView` summary, DESIGN.md § 7.1): left untyped.
