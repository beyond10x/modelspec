# Kernel IR, a fixed safe schedule and the execution plan of a training step, in modelspec-core

## Context

DESIGN.md § 3 puts two levels between the Training IR M0 builds and SPIR-V: Kernel IR ("low-level
loops, index expressions, dispatch dimensions, ... buffer spaces, layouts, barriers, and
lifetimes") and the Execution DAG ("dispatch and copy commands, resource dependencies/barriers,
scratch allocations, and CPU-visible completion points"). DESIGN.md § 5.3 fixes M1 to "a fixed
safe schedule", and § 7.1 has the compiler emit `CompiledArtifact { manifest, kernels, plan }`.
`spec/domains/artifact.yaml` types the artifact, its `PlanStep` (ops Allocate, Copy, Dispatch,
Barrier, Complete) and `BufferView`; it has no UNMAPPED marker, but it cannot yet say which buffer
view holds a parameter, an input, the loss or the prediction, so no runtime could bind them.
Nothing in the code builds either level: `crates/modelspec-core/src/compile.rs` stops at
`CompiledTraining`.

The plan this story fixes is what the 10x lane of `epic:pytorch-10x-mnist` needs from M1: one step
body that a runtime can record many times into one command buffer. So the step plan carries no
CPU-visible completion point; the runtime places one at the end of a run and at each requested
checkpoint. DESIGN.md § 6.3 (revision 0.9, the 10× patch this story is written against) item 3
has M2 keep two versions of each parameter, step s reading version s mod 2 and writing (s + 1) mod
2; the parameter type this story pins must hold that without a breaking change.

## What it adds

- `spec/domains/artifact.yaml`: `ParameterSpec.views: List<BufferView>` (one view is the in-place
  case: the step reads version n and the update writes n + 1 into the same view; two views are
  ping-pong, § 6.3 item 3; M1 emits one, M2 two), `DataSpec.view`, `CompiledArtifact.loss_view`,
  `CompiledArtifact.prediction_view` and `CompiledArtifact.forward_plan` (the inference plan: the
  forward pass without loss, gradients or update), and in the summary of `plan` that the step plan
  has no `Complete` step.
- `crates/modelspec-core/src/ir/kernel.rs`: the kernels of the training step, each with its kind
  (elementwise, broadcast, reduce stage 1, reduce stage 2, single-invocation loss sum, transpose,
  matmul, SGD update), shapes, dtype, bindings, workgroup size and dispatch dimensions; a printer
  with stable ids. The workgroup size of each kind is a compile-time constant of the schedule,
  never read from a DeviceProfile.
- `crates/modelspec-core/src/schedule.rs`: the fixed safe schedule. Ops in Training IR order; one
  Dispatch for each elementwise, broadcast, transpose, matmul (one invocation per output element)
  and SGD op; the loss sum of `mse` as one Dispatch of one invocation (row-major, the bit-for-bit
  class of § 5.4); every other reduce as a stage-1 Dispatch, an `Allocate` of a partials scratch
  buffer, a `Barrier` and a stage-2 Dispatch, in that order in the plan. A barrier wherever a
  dispatch reads what an earlier one wrote, one buffer per value (no reuse), and each SGD update in
  place, placed after every read of the version it replaces.
- `crates/modelspec-core/src/artifact.rs`: `CompiledArtifact` for `Backend::Cpu` and
  `Backend::Vulkan` with manifest, plan and forward plan; `source_hash`, `ir_hash` and
  `schema_hash` (blake3 over the ordered parameter paths, dtypes and shapes, independent of the
  backend and of the learning rate). Kernel module ids derive from the content hash of each kernel,
  so two compiles of one file give equal ids. A Cpu artifact has an empty plan and no kernels.
- `crates/modelspec-core/src/ir/verify.rs`: a plan verifier. Every dependency lies before its
  step, every view lies inside its buffer, each update depends on every reader of the version it
  overwrites, each stage-2 Dispatch reads a partials buffer an earlier `Allocate` made and depends
  on a `Barrier` after its stage 1, a parameter has one or two views, and workgroup sizes stay
  within the limits Vulkan guarantees on every device (128 invocations, 65,535 workgroups per
  dimension).
- `modelspec ir <file> --stage kernel`, printing the kernels and both plans.

## Domain relations

- CompiledArtifact owns KernelModule (`CompiledArtifact.kernels`); a Dispatch names its module by
  `PlanStep.kernel_module_id`. Inferable from ess/1.
- A parameter has one view across steps in M1: read at version n, written at n + 1
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
  `crates/modelspec-core/Cargo.toml` (`blake3`), `Cargo.lock`.
- `crates/modelspec-core/tests/kernel_plan.rs` (new),
  `crates/modelspec-core/tests/kernel_plan_adversary.rs` (new).
- `crates/modelspec-cli/src/ir.rs`, `crates/modelspec-cli/tests/ir_kernel.rs` (new),
  `crates/modelspec-cli/tests/golden/linear_regression.kernel.ir` (new).
- `crates/modelspec-docs/src/cli.rs`, and `website/docs/reference/cli.md` through `task
  docs-generate`. `story:observatory-live-event-stream` also regenerates `cli.md`; this story
  regenerates it on `main` after wave observatory-1 merges.
- `DESIGN.md` § 3.1 and § 7.1 (the views, the forward plan, the two-stage reduce in the plan, no
  completion point in the step plan). The revision line and `CHANGELOG.md` are append-only
  surfaces: this story adds its entry, and the wave coordinator assigns the revision number at
  merge.
- `CHANGELOG.md`.

