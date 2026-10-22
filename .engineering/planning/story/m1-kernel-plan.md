---
format: aep.planning-md/3
id: story:m1-kernel-plan
kind: story
status: active
title: Kernel IR, a fixed safe schedule and the execution plan of a training step, in modelspec-core
summary: Parameter, data, loss and prediction views and a forward plan in spec, Kernel IR with fixed workgroup sizes, a verified step plan with every reduce in it (one-invocation Mse loss sum, two stages around a partials buffer and a barrier otherwise) and no completion point, and modelspec ir --stage kernel
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:observatory-live-event-stream
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Cargo.lock
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
  path: website/docs/reference/cli.md
revision: 10
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T03:18:54Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"review_outcome":4}}}
- {from: "proposed", to: "active", at: "2026-10-22T03:18:54Z", actor: "human:timo", revision: 10, decided_on: {"recorded":{"review_outcome":4}}}
---
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
  never read from a DeviceProfile: 64 invocations for every kind but the loss sum, which has 1.
- `crates/modelspec-core/src/schedule.rs`: the fixed safe schedule. Ops in Training IR order; one
  Dispatch for each elementwise, broadcast, transpose, matmul (one invocation per output element)
  and SGD op; the loss sum of `mse` as one Dispatch of one invocation (row-major, the bit-for-bit
  class of § 5.4); every other reduce (`ReduceSum`, `ReduceMean`) as four plan steps in this
  order: an `Allocate` of its partials scratch buffer, a stage-1 Dispatch that writes one partial
  per workgroup into it, a `Barrier`, and a stage-2 Dispatch of one workgroup that reads the
  partials (the stage-1 Dispatch writes the partials, so the `Allocate` comes first). An
  `Allocate` is a run-setup step: a runner executes every `Allocate` of a plan once, before its
  first recording, and records only the other steps per training step. The verifier refuses an
  `Allocate` whose buffer an earlier step uses, so moving it to run setup changes no result.
  DESIGN.md § 7.1 states this rule in the same change. A barrier wherever a
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

## Acceptance

The compiler turns every M0 training step into verified Kernel IR and a plan whose views a runtime
can bind, with every reduce inside the verified plan (one single-invocation Dispatch for the `mse`
loss sum, two stages around a partials buffer and a barrier for any other reduce), no completion
point inside the step, and `ir --stage kernel` shows both plans.

1. First: `spec/domains/artifact.yaml` types `ParameterSpec.views` (one or two BufferViews, stated
   as an invariant), `DataSpec.view`, `CompiledArtifact.loss_view`,
   `CompiledArtifact.prediction_view` and `CompiledArtifact.forward_plan`, and `ess specify
   validate --path .`, run from inside `spec/`, prints `valid` with no new warning.
2. `cargo test -p modelspec-core --test kernel_plan`: for `examples/linear_regression.mspec` and
   `crates/modelspec-docs/models/two_layer_relu.mspec` the step plan has exactly one Dispatch per
   computing op of the training step (every op but Input, ParamRead and Scalar), the `Mse` Dispatch
   has workgroups `[1, 1, 1]` and workgroup size 1, every parameter has exactly one view, there is
   no `Complete` step, and the plan passes the verifier; the forward plan writes no parameter view
   and has no SGD dispatch.
3. The same test, on a test-local source whose loss is `mean(mul(model.forward(x), y))`: its
   `ReduceMean` appears as exactly four consecutive plan steps, `Allocate`, stage-1 Dispatch,
   `Barrier`, stage-2 Dispatch; the stage-1 Dispatch writes the view the `Allocate` made, the
   stage-2 Dispatch reads it and depends on the `Barrier`, and the plan passes the verifier. Every
   other computing op still has exactly one Dispatch.
4. `cargo test -p modelspec-core --test kernel_plan_adversary`: the verifier refuses each of ten
   altered plans with its own error: a dependency on a later step, an update that does not depend
   on a reader of its parameter, a view past the end of its buffer, a workgroup of 129
   invocations, 65,536 workgroups in one dimension, a Dispatch without a kernel module, a stage-2
   Dispatch whose partials buffer no earlier `Allocate` made, a stage-2 Dispatch with no `Barrier`
   after its stage 1, a parameter with no view, and a parameter with three views.
5. Two compiles of one file give byte-equal printed Kernel IR and equal kernel module ids;
   `schema_hash` is equal for Cpu and Vulkan artifacts and for two learning rates, and differs when
   one parameter shape changes. The entry point of `schedule.rs` takes no DeviceProfile, and the
   printed Kernel IR shows each kernel workgroup size as the constant of its kind.
6. `cargo test -p modelspec-cli --test ir_kernel`: `modelspec ir examples/linear_regression.mspec
   --stage kernel` exits 0 and equals `crates/modelspec-cli/tests/golden/linear_regression.kernel.ir`
   byte for byte; `--stage kernels` exits 2.
7. `task ci` and `task core-no-vulkan` exit 0; `modelspec-core` gains no Vulkan, SPIR-V or GPU
   dependency.

## Out of scope

- SPIR-V emission and the artifact on disk: `story:m1-vulkan-elementwise`,
  `story:m1-vulkan-reduce-matmul`, `story:m1-compile-artifact`.
- Two views per parameter (ping-pong, DESIGN.md § 6.3 item 3): the type admits them, the verifier
  accepts them, and no M1 schedule emits them (M2, `epic:pytorch-10x-mnist`).
- Fusion, tiling, liveness-based buffer reuse and a whole-run schedule (M2,
  `epic:pytorch-10x-mnist`).
- Workgroup sizes chosen from a DeviceProfile (M2 measures alternatives, DESIGN.md § 5.3).
- Reductions over one axis of a higher-rank tensor: no M1 example needs one.
- Device ownership of a buffer view (`BufferView` summary, DESIGN.md § 7.1): left untyped.
