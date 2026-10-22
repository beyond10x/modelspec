# ModelSpec — Design Specification

**Status:** M0 implemented (iteration 1); M1 onwards proposed  
**Revision:** 0.N (2026-10-22): the M1 kernel plan, with buffer views, a step plan and a forward plan that end with their one fence instead of a completion point, two-stage reductions, and E-TARGET-010 for a tensor too large to dispatch or bind (§ 3.1, § 7.1, § 8.1); 0.9: ADR-007 (proposed) gates 10× PyTorch eager throughput on the MNIST benchmark example on the RTX 3090, with the whole-run schedule that reaches it and the GPU results compared bit for bit (§ 1, § 4, § 5.2, § 5.4, § 6, § 6.1, § 6.3, § 9, § 12); 0.8: the input adjoint, the gradient of a rank-0 target with respect to named inputs in its own program (§ 3.3 pass 4, § 4); 0.7: M3 adds a small BERT-like masked-language-model proof model beside the GPT-like one (§ 1, § 11 iteration 4); 0.6: derivation of parameter, teacher and synthetic-data draws defined (§ 2.3), M0 `train` output and codes (§ 8, § 8.1); 0.5: a train loss calls `model.forward` exactly once (§ 2.1)  
**Implementation language:** Rust  
**Initial execution target:** NVIDIA RTX 3090 via Vulkan compute / SPIR-V  
**Principle:** Specify *what* to compute; the compiler determines *how* to execute it.

## 0. Executive decision

ModelSpec is a versioned, declarative language and compiler for specifying differentiable tensor programs. A model is built compositionally from scalar functions, tensor operations, reusable modules, and architectures. A training specification adds a loss, optimizer, data bindings, and execution policy. The compiler elaborates the entire training step, generates its gradient computation, optimizes the resulting graph, schedules GPU kernels, and runs them in an independently owned Rust runtime.

The initial stack has **no Python, PyTorch, or CUDA API/toolkit dependency**. It uses the system Vulkan loader and an installed NVIDIA Vulkan driver, which translates the emitted SPIR-V into GPU-native instructions. This is **not** direct generation of NVIDIA SASS or an attempt to replace the NVIDIA kernel-mode driver.

We optimize for correctness and coherent architecture before raw speed. Near-optimal performance on a wide range of neural networks is a long-term research and engineering goal, **not** a v1 guarantee.

### Non-negotiable invariants

1. **One semantic core.** CPU reference execution and GPU compilation consume the same verified tensor IR.
2. **Composition does not mandate kernel boundaries.** Functions/layers are inlined or lowered; operations may fuse or split.
3. **Shapes and dtypes are explicit.** The compiler rejects invalid operations before dispatch.
4. **Purity by default.** Parameters, random state, and optimizer updates have explicit identities/effects; graph rewrites cannot silently change them.
5. **Training is compiled.** Forward, loss, backward, and optimizer are part of a single optimizable `train_step` program.
6. **Hardware features are queried.** No assumptions that a particular Vulkan extension, dtype, or cooperative matrix shape is supported by the driver's current configuration.
7. **Testable at every lowering stage.** Introspection, differential testing, and profiling are design requirements.

---

## 1. Scope and milestones

| Release | Deliverable | Not yet required |
| --- | --- | --- |
| **M0: semantic core** | Parse, elaborate, type/shape check, typed SSA tensor IR, CPU reference interpreter, reverse-mode autodiff, SGD, simple linear/MLP training | GPU or performance |
| **M1: native Vulkan execution** | Rust Vulkan device discovery/allocator/dispatch, SPIR-V elementwise + reduction + basic matmul kernels, compiled forward/backward/SGD, checkpoints | Tensor Core speed |
| **M2: optimizing compiler** | Whole-step liveness, buffer reuse, elementwise fusion, tiling/shared-memory matmul, kernel scheduler, profiling + cache, whole-run execution with data and weights resident on the device (§ 6.3), the throughput gate of ADR-007 | All tensor ops |
| **M3: neural-network essentials** | f16 where supported, AdamW, softmax, layer norm, embeddings, attention, causal and bidirectional masks, GELU, small GPT-like training and small BERT-like masked-language-model pre-training | Large-scale distributed training |
| **M4: device specialization** | Cooperative-matrix Tensor Core path where available, mixed precision with FP32 accumulation/master state when needed, checkpoint/recompute planning, autotuning | Universal peak performance |

M0 should be independently useful; M1 is the first milestone meeting the end-to-end RTX 3090 training goal. Everything above M1 is incremental.

### Initial deliberate exclusions

Distributed training, dynamic Python-like objects, implicit eager execution, arbitrary user-authored GPU assembly, arbitrary runtime recursion, variable-length control flow, graph mutation during execution, production data-loader ecosystem, arbitrary-stride views, automatic cross-device partitioning, and direct SASS emission. These may be considered later; they must not distort M0/M1 design.

---

## 2. ModelSpec source language

Source files end in `.mspec`. The language is *declarative, statically typed, pure by default*, with generic dimension parameters and explicit state declarations. Modules are reusable units. `forward` is a regular pure function, not a magical runtime hook.

```modelspec
module Linear<In: dim, Out: dim> {
    param weight: Tensor<f32, [In, Out]> = xavier_uniform();

    fn forward<B: dim>(x: Tensor<f32, [B, In]>)
        -> Tensor<f32, [B, Out]> {
        return matmul(x, weight);
    }
}

model TinyMLP<B: dim> {
    layer fc1: Linear<32, 64>;
    layer fc2: Linear<64, 1>;

    fn forward(x: Tensor<f32, [B, 32]>)
        -> Tensor<f32, [B, 1]> {
        let h = relu(fc1.forward(x));
        return fc2.forward(h);
    }
}

train TinyFit {
    model = TinyMLP<64>;
    input  x: Tensor<f32, [64, 32]>;
    target y: Tensor<f32, [64, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 1000;
    seed = 42;
    device = vulkan(0);
}
```

This is the **target language design**, not a claim that a parser already exists. M0 can start with a restricted subset: one model, static dimensions, `f32`, parameters, local `let`, `matmul`, `add`, `sub`, `mul`, `relu`, `mean`, `mse`, and SGD. Introduce modules and generics through compile-time elaboration without changing core tensor semantics.

### 2.1 Surface constructs

| Construct | Meaning | Lowering |
| --- | --- | --- |
| `fn` | Pure, typed mathematical function | Inline or call graph specialization |
| `module` | Reusable collection of params/functions | Instance elaboration with stable names |
| `layer` | Named module instance | Statically instantiate; no runtime object |
| `model` | Parameterized architecture | Closed graph from entry-point functions |
| `param` | Persistent named trainable tensor | Parameter handle + initialization spec |
| `const` | Immutable scalar or tensor | Constant IR value |
| `train` | Loss, optimizer, dataset interface, execution policy | Stateful training-step IR |
| `infer` (later) | Inference signature and export policy | Read-only compiled execution graph |
| `target` (later) | Optional capabilities/optimization constraints | Backend configuration, not model semantics |

The loss of a `train` block calls `model.forward` exactly once, counting calls nested in its arguments; the checker refuses zero calls at the loss and a second call at that call (E-TYPE-017).

### 2.2 Type system and shapes

Canonical tensor type: `Tensor<dtype, [shape...]>`, where each dimension is a positive compile-time integer or a generic dimension that must be resolved for M0/M1 compilation. Rank 0 means scalar. `f32` is the only guaranteed M0/M1 numeric dtype. Later add `f16`, signed integers, boolean masks, and quantized formats behind verified device capabilities.

M0/M1 shape rules, intentionally strict:

- `matmul([M,K], [K,N]) -> [M,N]` for rank-2 tensors; mismatches are compile errors.
- `add`, `sub`, `mul`: same-shape inputs, or a scalar and tensor; **no implicit non-scalar broadcasting yet**.
- `relu([*S]) -> [*S]`; `transpose([M,N]) -> [N,M]`.
- `sum([*S]) -> []`, `mean([*S]) -> []` with an explicit specified reduction order policy.
- `mse(a, b) -> []` iff `a` and `b` have identical shapes; defined as `sum((a-b)^2) / element_count(a)`.
- Reduction order policy (the CPU reference, § 9 gate 2): a sum accumulates sequentially in row-major index order in the element type, starting from its first term, with every product, difference and partial sum rounded to the element type (no fused multiply-add); `matmul` accumulates each output element over the contracting index in ascending order the same way; `mean` is that sum divided by the element count converted to the element type; `mse` is that sum of `(a-b)^2` divided by the element count converted to the element type.
- Empty shapes, negative dimension sizes, integer overflows in element/byte counts, and allocations over backend limits are hard errors.

All built-in operators define forward values, gradient rules, shape constraints, precision behavior, and side effects in one operator registry (see §4). No implicit CPU fallback from a GPU-compiled plan without an explicit error or permitted fallback policy.

### 2.3 Parameters, sharing, and randomness

- Each parameter has a **stable canonical path**, e.g. `TinyMLP.fc1.weight`, independent of optimization order.
- Two `Linear` instances own distinct weights. Explicit `share(a.weight)` (post-M1) is required for tied weights; accidental aliasing is disallowed.
- Parameters have explicit initialization semantics; initial values are generated reproducibly from `global_seed + canonical_parameter_id` via the counter-based PRNG Philox4x32-10 (Salmon et al., SC 2011), **not** via compiler iteration order. `canonical_parameter_id` is the first 8 bytes of the `blake3` hash of the canonical path, read as a little-endian `u64`.
- Derivation. A draw is addressed by key and counter alone. The key is `[seed as u32, (seed >> 32) as u32]` from the `train` block's `seed`. The counter is `[index, stream, id as u32, (id >> 32) as u32]`: the element index in row-major order, a stream tag, and the 64-bit id of what is drawn. The id is `canonical_parameter_id` for a parameter and the same `blake3` id of its name for a data input. The stream tags are `0` for initial parameters, `1` for the teacher parameters of M0 synthetic data (§ 9 gate 4) and `2` for M0 synthetic inputs. Element `i` takes the block `philox4x32_10(counter(i), key)`. A uniform draw maps word 0 to `u = (w + 0.5) / 2^32` in `(0, 1)`, and `xavier_uniform` of a `[In, Out]` parameter is `a * (2u - 1)` with `a = sqrt(6 / (In + Out))`. A standard normal draw is Box-Muller over words 0 and 1, `sqrt(-2 ln u0) * cos(2π u1)`. Each value is computed in `f64` and rounded to `f32`. A tensor with more than `2^32` elements, or `xavier_uniform` on a tensor that is not rank 2, is refused.
- Randomness used in a forward pass (e.g. dropout, post-M2) is an explicit RNG state/token. This prevents common-subexpression elimination from incorrectly merging independent random draws.
- A model specification does not do I/O. Dataset and checkpoint I/O belongs to the runtime or training host.

### 2.4 Control flow and dynamism

M0/M1 compile-time constructs (`repeat`, module instantiation, dimension generics) **elaborate statically**. Runtime data-dependent `if`/`while` and dynamically ragged tensors are out of scope. Bounded dynamic shapes may be added through specialization and recompilation. The compiler will cap specialization count to avoid unbounded code growth.

---

## 3. Compilation pipeline

```text
modelspec source (.mspec)
  -> lexer / parser                   [Source AST]
  -> name resolution + module expand  [Elaborated AST]
  -> tensor type/shape/effect checks  [Typed HIR]
  -> tensor SSA graph                 [Tensor IR]
  -> canonicalize, fold, DCE
  -> differentiate + optimizer step   [Training IR]
  -> whole-step optimize + liveness   [Optimized Training IR]
  -> partition/fuse + schedule/tiles  [Kernel IR + Execution DAG]
  -> target lowering                 [SPIR-V modules + manifest]
  -> Vulkan pipeline creation        [driver produces native GPU code]
  -> Rust-owned execution/runtime    [train, checkpoint, infer]
```

Passes operate on explicit immutable IR values; state transitions and synchronization are represented explicitly. Every pass has preconditions and postconditions, and the verifier runs after major transformations in debug/test builds.

### 3.1 IR levels

**Typed HIR:** source-level functions/modules with resolved types, dimension bindings, spans, and stable parameter paths. No execution layouts.

**Tensor IR:** typed SSA values with semantic ops such as `MatMul`, `Relu`, `ReduceMean`, `Transpose`, `ParameterRead`. Graph is referentially transparent except for explicit effects. Records layout constraints, not a particular layout choice. Every tensor value has a shape, dtype, and defining operation.

**Training IR:** explicit `Forward`, `Loss`, `Gradient`, `Optimizer` subgraphs for analysis, but **not** mandatory execution boundaries. Reads and writes to persistent parameter/optimizer state carry ordered effect tokens. A parameter update produces a new version of a state handle; no optimizer pass is allowed to overwrite data still required by backward computation.

**Kernel IR:** low-level loops, index expressions, dispatch dimensions, workgroup/subgroup mappings, buffer spaces, layouts, barriers, and lifetimes. A single kernel may implement multiple tensor ops, and one tensor op may require multiple kernels.

**Execution DAG:** dispatch and copy commands, resource dependencies/barriers, scratch allocations, and CPU-visible completion points. The runtime executes the DAG, not the original model AST.

M1 schedules the Kernel IR and the Execution DAG together (`schedule::schedule`, printed by `modelspec ir --stage kernel`): one buffer and one view per value and per parameter, except that the SGD update writes its parameter's buffer and view in place rather than a buffer of its own, scalars compiled into kernels as immediates, one dispatch per computing op, and a full barrier on every read-after-write, write-after-read or write-after-write hazard. A full reduction is four steps: an `Allocate` of its partials buffer, a stage-1 dispatch, a barrier and a stage-2 dispatch. The artifact carries two plans: the step plan, and a forward plan that computes the prediction into its own view and writes no parameter view; a forward that returns an input or a parameter copies it there (a `Copy` step), and one that returns a literal fills it. Neither plan has a completion point: each has exactly one fence, a `Barrier` with no dependencies that is ordered after every earlier step, and it is the plan's last step; `verify_plan` refuses a fence anywhere else. A runner may record a plan back to back as many times as it chooses, the recordings concatenated, each after the fence of the one before, and places completion points itself; `verify_recorded_back_to_back` checks that concatenation as one plan in which each fence closes a recording and a step with no dependencies after a fence is ordered after it. The runner uploads each batch between recordings, outside the recorded plan. Within one recording, `verify_plan` refuses a read of any parameter after its in-place update, and a dispatch whose workgroups differ from its kernel's; an `Allocate` is run setup, so it refuses one whose buffer an earlier step of the same recording used. No data, loss or prediction view lies in a parameter buffer, and each parameter and data view holds its declared type's element count; the loss view holds one element and the step plan writes it, and both plans write the prediction view. Workgroup sizes are per-kernel-kind constants (64; 1 for the loss sum), not read from a device profile.

### 3.2 Suggested Rust representations (illustrative)

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType { F32 } // F16, etc. added once supported

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim { Static(u64), Symbol(u32) }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<Dim>,
}

pub enum TensorOp {
    Input { name: String },
    ParamRead { param: ParamId },
    Scalar(f32),
    Add, Sub, Mul, MatMul, Relu,
    Transpose,
    ReduceSum, ReduceMean,
}

pub struct TensorNode {
    pub id: ValueId,
    pub op: TensorOp,
    pub operands: Vec<ValueId>,
    pub ty: TensorType,
    pub source_span: Option<SourceSpan>,
}

// Exact Rust traits are an implementation detail to refine during M0.
// Preserve a target-agnostic compilation boundary:
pub trait KernelBackend {
    type Executable;
    type Error;
    fn compile(&self, program: &ScheduledProgram)
        -> Result<Self::Executable, Self::Error>;
}
```

Use *typed IDs* rather than vector positions in public APIs. The verifier checks SSA dominance/topological definition order, ownership of parameter IDs, legal effects, shape/dtype consistency, and full output initialization.

### 3.3 Compiler passes and ordering

1. Parse, resolve names, specialize compile-time dimension parameters.
2. Verify tensor shapes/dtypes/effects; lower functions/layers to tensor graph.
3. Canonicalization, constant propagation, algebraic simplification where IEEE floating-point rules allow, dead-code elimination.
4. Expand the selected loss to a differentiable graph; reverse-mode autodiff over only loss-reachable trainable values. The input adjoint, the gradient of any rank-0 target with respect to named inputs, is the same walk in a separate program (§ 4); the training program does not change.
5. Attach explicit optimizer-state computations and ordered parameter updates.
6. Graph-wide dead-code elimination; forward/backward fusion candidates; liveness and recomputation analysis.
7. Choose layout, partition into kernels, tile work, plan buffer lifetimes; **repeat** some optimization after layout/scheduling.
8. Generate and validate SPIR-V, serialize execution manifest, instantiate Vulkan pipelines.

Do not silently assume `a+b == b+a` at bit level, or reassociate floating-point reductions in deterministic mode. `fast_math = true` is a distinct opt-in compilation setting that is recorded in the artifact hash. `deterministic` is a compilation setting too: it fixes reduction order, so it is recorded in the artifact hash, and a run reads it from its artifact.

---

## 4. Automatic differentiation

Use **reverse-mode AD on tensor SSA**, not tape-based Python autograd. One backward graph is generated when the `train` specification is compiled. A small, explicit gradient-rule registry is a first-class compiler component.

For output adjoint `G = dL/dY`:

| Forward op | Reverse-mode contribution |
| --- | --- |
| `Y = A + B` | `dA += G`; `dB += G` |
| `Y = A - B` | `dA += G`; `dB -= G` |
| `Y = A * B` | `dA += G * B`; `dB += G * A` |
| `Y = A @ B` | `dA += G @ transpose(B)`; `dB += transpose(A) @ G` |
| `Y = relu(A)` | `dA += where(A > 0, G, 0)`; derivative at 0 **defined as 0** |
| `Y = mean(A)` | `dA += broadcast_scalar(G / numel(A), shape(A))` |
| `L = mse(Y,T)` | `dY = 2 * (Y-T) / numel(Y)`; targets receive no gradient |

`transpose`, `where`, and scalar broadcasting are explicit internal gradient ops, even if the user-facing M0 language keeps simpler syntax. For shared graph subexpressions, **accumulate** every incoming adjoint exactly once. Treat parameters and trainable module state separately from temporary SSA tensors.

**Input adjoint.** The input adjoint is the gradient of any rank-0 target with respect to named inputs instead of parameters, by the same walk and the same rules (`differentiate_inputs`). The target is the loss, or one element of the prediction built from existing ops as `sum(prediction * target_mask)`, where `target_mask` is a new input of the prediction's type that the caller binds to the one-hot tensor at the element's row-major offset (`output_element_target`). The result is a separate verified program whose outputs are the target, then the adjoint of each named input with that input's type; the training program, its outputs and what `ir --stage training` prints do not change. An input the target does not read is not refused: its adjoint is zero, `broadcast_scalar(0, shape)`. The `mse` target receives `-dY` only when it is itself named. Refused: a name that is not an input of the program (a declared input that dead-code elimination removed has none), a target that is not rank 0, a prediction that is not a value of the program, an index of the wrong rank or out of bounds, and a program that already has an input named `target_mask`.

Initial strategy: retain forward intermediates required by AD; introduce recomputation/checkpoint planning after correctness is proven. Training state writes happen only after all old-state reads are complete. Gradients accumulate in `f32` for M0/M1.

**Gradient verification:** compare CPU reverse-mode gradients to central finite differences evaluated in `f64` reference math, away from non-differentiable points. Check selected tiny graphs under seeded inputs. GPU results are checked within documented absolute/relative tolerances; § 5.4 names the results that must in addition equal the CPU oracle bit for bit.

---

## 5. GPU kernel compiler

### 5.1 Initial backend: Vulkan / SPIR-V

Use Rust `ash` for Vulkan API access and `rspirv` to construct SPIR-V modules programmatically. Avoid a necessary GLSL/HLSL compiler in the production compilation path. For bring-up or diagnosing unexpected SPIR-V behavior, reference GLSL and external `spirv-val` may be used as development tools.

The generated SPIR-V is **shader IR**, not the physical Ampere instruction set. The installed NVIDIA Vulkan driver owns the final machine-code compilation, internal instruction selection, and GPU submission. The compiler controls tensor-level scheduling and shader construction, not undisclosed driver internals.

Device discovery must record: vendor/device IDs, device UUID if available, driver version, SPIR-V/Vulkan feature levels, subgroup properties, FP16/storage features, cooperative matrix extensions/properties, workgroup and storage-buffer limits, timestamp support, and memory heaps/budgets. Failure to support a specialized path triggers a **correct fallback** or explicit unsupported-feature diagnostic.

### 5.2 Kernel progression

**M1 reference GPU kernels**

- Elementwise: flat-index global dispatch, explicit bounds checks, `f32` loads/stores.
- Reductions: 2-stage reduction (workgroup partials then final), with numerical behavior documented.
- Matmul: straightforward bounds-safe tiled or naïve `f32` shader to establish correctness, **not** claimed as performant.
- Transpose and scalar broadcast: separate kernels or fuse into consumers.
- SGD: `p_out = p_in - lr * gradient`, respecting parameter versioning, using in-place update only after liveness proves it safe.

**M2 optimized kernels**

- Workgroup tiling and shared-memory staging for matmul; choose tile size to fit shared memory and register pressure. In `deterministic` mode one invocation accumulates each output element over the contracting index in ascending order, tile after tile, and the contracting index is not split across invocations (no split-K), so the order of § 2.2 holds.
- Fuse elementwise chains and simple epilogues (e.g. `matmul + relu`) if compiler cost estimates permit. For a training step this includes `relu` into the loads of its consumers, the MSE gradient `(Y - T) * c` into the matmul that produces `Y`, `ReluGrad` into the matmul that produces its adjoint, and the SGD update `p - lr * g` into the matmul that produces `g`, so `g` is not written to device memory.
- Gather operands in the matmul load: minibatch rows are read from the resident data set through the resident permutation (§ 6.3), with no separate minibatch copy.
- Plan layouts and alignment, avoid unnecessary global writes, batch compatible kernels.
- Tune with GPU timestamp queries; measure registers/occupancy via external profiling where practical.

**M4 hardware specialization**

- Probe `VK_KHR_cooperative_matrix` or applicable vendor extension at runtime; enumerate *actual supported* matrix types and M/N/K shapes.
- Dispatch cooperative-matrix tiled matmul only on devices with a validated implementation. Do not assume a specific driver exposes this extension merely because the GPU has Tensor Cores.
- Incorporate f16 storage/compute with explicit accumulation and stability rules. Do not assume bfloat16 is available in the initial Vulkan path.

### 5.3 Fusing and partitioning

Fusion is cost-driven, not automatic: fewer launches and intermediate writes may lose to increased registers, workgroup occupancy, or duplicate computation. Kernel partitioning considers memory traffic, tiling compatibility, operation shapes, shared-memory footprints, register estimates, and dependency barriers.

M1 uses a fixed safe schedule; M2 measures alternatives; M4 supports offline/autotuned specialization. Kernel configuration cache keys include IR hash, shapes, dtype, optimization settings, GPU/vendor/driver identifiers, and extension/capability set.

### 5.4 GPU results compared bit for bit and within tolerance

The CPU oracle is the reference for every GPU result (§ 9 gate 5). Which comparison applies is fixed by the instructions that produce a result and by the device's queried features:

1. **Bit for bit.** In `deterministic` mode, on a device that reports `shaderRoundingModeRTEFloat32`, `shaderDenormPreserveFloat32` and `shaderSignedZeroInfNanPreserveFloat32`, kernels declare the execution modes `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` and decorate every `OpFAdd`, `OpFSub` and `OpFMul` with `NoContraction`. Vulkan specifies these three instructions as correctly rounded and comparisons as exact (Reference 1); `NoContraction` forbids contracting them into a fused multiply-add and rearranging them. A result built only from these instructions, copies and selects, in the order of § 2.2, equals the CPU oracle bit for bit. For a training step these are: the gathered minibatch and its one-hot targets, every `matmul` output without split-K, `relu` and `ReluGrad` as compare and select, `sub`, `mul` by the gradient scale of `mse` (folded to a constant at compile time), the SGD update, the parameters after every step, and a loss sum accumulated by one invocation in row-major order.
2. **Within tolerance.** Every other result is compared within the tolerances of § 9 gate 5: a result that passes through `OpFDiv` (Vulkan bounds it at 2.5 ULP; in a step only the `mse` and `mean` values, which no gradient reads), a transcendental function, any parallel or split reduction, any `fast_math` artifact, and every result on a device that lacks one of the three features. The run record names the class the run used.
3. A bit-for-bit result also passes the tolerance check: this adds to gate 5 and replaces nothing in § 9. Cooperative-matrix kernels (§ 5.2, M4) are never in the bit-for-bit class, and the M2 schedule does not use them.

---

## 6. Rust runtime design

The runtime has four owned responsibilities:

**Device:** Vulkan instance/device/compute queue selection, feature negotiation, shader modules, pipeline cache, command pools, and error reporting. Prefer a compute-only headless configuration; no graphics swapchain.

**Memory:** device-local tensor arena, staging buffers for host upload/download, allocations obeying Vulkan heap/buffer-range/alignment restrictions, liveness-based scratch-buffer reuse, bounds/overflow checks, and out-of-memory diagnostics. Keep checkpoint and optimizer state in separate logical arenas even if physically suballocated.

**Execution:** execute compiled DAG nodes (uploads, dispatches, barriers, reads); construct descriptor resources, push constants, specialization constants, command buffers, and synchronization. Use appropriate transfer/compute barriers (e.g. synchronization2 when available). No hidden host synchronization between every operator; initial implementation may block only at train-step completion to simplify correctness. From M2 the runtime does not block per step either (§ 6.3).

**Training host:** data batching interface, seed/state initialization, step counter, loss telemetry, checkpoint save/reload, and inference entry point. Host is Rust; model does not invoke arbitrary host callbacks in M1.

Proposed API boundary:

```rust
pub struct TrainConfig {
    pub steps: u64,
    pub seed: u64,
    pub checkpoint_every: Option<u64>,
}

pub trait DeviceRuntime {
    type Compiled;
    type Error;
    fn load(&mut self, artifact: &CompiledArtifact)
        -> Result<Self::Compiled, Self::Error>;
    fn execute_step(&mut self, program: &mut Self::Compiled,
                    inputs: &Batch) -> Result<StepMetrics, Self::Error>;
}
```

`steps` counts the steps of one invocation, so a run resumed at step `s` ends at `s + steps`. This interface is illustrative; use opaque handles rather than exposing raw Vulkan objects to the training host. The CPU backend implements equivalent observable training semantics for debugging.

### 6.1 Reproducibility

Record source/compiler version, tensor shapes, optimizer hyperparameters, RNG algorithm/seed/state, model/parameter identifiers, step number, backend/device/driver, and compile flags. A seeded training run should be reproducible **under a documented fixed configuration**, not promised bit-identical across CPU and GPU or different drivers, except for the results § 5.4 compares bit for bit on a device that reports the features it names. `deterministic` mode (a compile flag, § 3.3) disables nondeterministic algorithms where controllable, but driver-dependent floating-point behavior still applies.

### 6.2 Checkpoint format

Versioned binary package `.msckpt`:

- header: magic `MSCK`, format version, endian marker, manifest size, checksum/hash (`blake3` over the header and manifest bytes, the checksum field itself counted as zero bytes);
- structured manifest (JSON for M1): compiler/model schema hash, parameter names, shapes, dtypes, offsets/lengths, optimizer state descriptors, RNG and step state;
- contiguous little-endian tensor payload with explicit alignment;
- optional verification hash of payload.

Tensor offsets are absolute file offsets. Validate offset and size ranges against file size before mapping/allocating. Load into CPU backend even if checkpoint was made on GPU. Parameter-path migration between incompatible model definitions requires an explicit adapter; never silently reinterpret tensor bytes.

### 6.3 Whole-run execution (M2)

M2 executes a training run, or a requested number of its steps, with no host round-trip per operator or per step. It changes the schedule only: every kernel computes the ops of the verified training IR (§ 0 invariants 1 and 5), and § 5.4 says which results equal the CPU oracle bit for bit.

1. **Resident state.** The data set, its labels, the permutation of every epoch the run covers, two versions of each parameter and one loss slot per reported step are uploaded once before the first step and stay in device-local memory; the allocator checks them against the heap budget (§ 6, Memory). For the MNIST benchmark example that is 188,160,000 bytes of images and 720,000 bytes of permutation. The upload is loading, outside the timed region of ADR-007.
2. **Gather on the device.** Step `s` reads its minibatch rows through `permutation[s * B + r]` in the operand load of its first matmul (§ 5.2), and builds its one-hot targets from the labels the same way. The permutation is the shuffle order of the data binding, computed on the host. The gather is a runtime data binding, not a Tensor IR op (§ 2.3: a model does no I/O).
3. **Parameter versions.** Step `s` reads version `s mod 2` of each parameter and writes version `(s + 1) mod 2` (§ 3.1). Two buffers replace an in-place update because the kernel that updates `W2` runs beside the kernel that still reads the old `W2`.
4. **Recording.** The host records the steps in chunks, one command buffer per chunk, with a compute-to-compute memory barrier between dependent dispatches and the step index in a push constant. It records chunk `n + 1` while the device executes chunk `n` and submits it to the same queue. The host waits only at the end of the run, at a checkpoint (§ 6.2) and at a requested read-back; a checkpoint ends a chunk. Steps are not merged into one dispatch behind a device-wide barrier, because Vulkan does not guarantee that the workgroups of one dispatch run at the same time.
5. **Read-back.** Parameters are read back at the end and at each checkpoint. On a reported step (§ 8) the loss is written to its slot and read back with the parameters; the `step <k> loss <v>` lines keep their content and order and are printed after that read-back. Per-kernel timing uses timestamp queries on sampled steps (§ 0 invariant 7).
6. **Schedule of the benchmark example.** Four dispatches per step, each a tiled shared-memory kernel (§ 5.2) with its elementwise ops fused. `X_b` and `T_b` are gathered as in item 2; `c` is the compile-time gradient scale of `mse`.

   | kernel | computes | writes |
   | --- | --- | --- |
   | K1 | `A = X_b @ W1` | `A` |
   | K2 | `dY = (relu(A) @ W2 - T_b) * c`; on a reported step also the loss sum | `dY`, loss slot |
   | K3 | `dA = ReluGrad(A, dY @ transpose(W2))` and, in other workgroups, `W2' = W2 - lr * (transpose(relu(A)) @ dY)` | `dA`, `W2'` |
   | K4 | `W1' = W1 - lr * (transpose(X_b) @ dA)` | `W1'` |

   If the four dispatch boundaries cost more than their budget, every K4 workgroup recomputes `Y`, `dY` and its columns of `dA` itself (at most 0.256 MFLOP per workgroup, from the shapes), one workgroup also writes `W2'`, and a step has two dispatches. The same ops in the same order keep § 5.4's class; § 5.3 decides between the two schedules by measurement.
7. **Budget.** ADR-007's gate is 51.9 µs per step. The work column is arithmetic from the shapes; every µs figure other than the gate is an estimate, not a measurement, and the first M2 benchmark replaces it.

   | part of a step | work per step | budget, µs |
   | --- | --- | --- |
   | K1 | 10.04 MFLOP; reads 313,600 B of `X_b` and 200,704 B of `W1` | 12 (estimate) |
   | K2 | 0.13 MFLOP | 3 (estimate) |
   | K3 | 0.26 MFLOP | 3 (estimate) |
   | K4 | 10.04 MFLOP; reads 313,600 B of `X_b`, writes 200,704 B of `W1'` | 12 (estimate) |
   | four dependent dispatch boundaries | barrier, drain and launch | 12, 3 each (estimate) |
   | host recording, submission and waits not hidden behind device work | | 2 (estimate) |
   | loss sum on reported steps | 1,000 terms on about 37 of 1,800 steps | 0.5 amortized (estimate) |
   | read-back at the end | 203,264 B of parameters and the loss slots | 0.5 amortized (estimate) |
   | reserve | | 6.9 |
   | total | 20.45 MFLOP | 51.9 (gate) |

   Derived from the table: the gate needs 394 GFLOP/s over a whole step, and K1 and K4 at 12 µs each run at about 0.84 TFLOP/s. Without fused multiply-add (§ 5.4) a multiply and an add are two instructions, so the usable `f32` peak is about half of the device's fused figure, about 17.8 TFLOP/s (estimate: half the vendor's 35.6 TFLOP/s, not read from a file). Without split-K each of K1's 6,400 output elements is one chain of 784 dependent additions, about 1.8 µs at 4 cycles per addition and 1.7 GHz (estimate). The dispatch boundaries are the largest unmeasured item; M2 measures a chain of empty dependent dispatches in one command buffer before it writes the kernels.

---

## 7. Rust workspace layout

Start as a small **workspace of three crates**, using modules for detail. Splitting everything into dozens of crates on day one would add unnecessary friction.

```text
modelspec/
  DESIGN.md
  Cargo.toml                     # workspace
  crates/
    modelspec-core/              # pure compiler; no Vulkan dependency
      src/
        lib.rs
        syntax/{lexer,parser,ast}.rs
        resolve.rs
        types.rs
        hir.rs
        ir/{tensor,training,kernel,verify,print}.rs
        autodiff.rs
        passes/{canonicalize,dce,fuse,liveness}.rs
        schedule.rs
        compile.rs
    modelspec-runtime/           # CPU oracle + Vulkan backend
      src/
        lib.rs
        cpu/{tensor,interpreter}.rs
        vulkan/{device,features,memory,pipeline,dispatch,sync}.rs
        spirv/{emit,validate}.rs
        train/{optimizer,runner,checkpoint,data}.rs
    modelspec-cli/               # CLI and user-facing errors
      src/main.rs
  examples/
    linear_regression.mspec
    tiny_mlp.mspec
  tests/
    language/
    gradients/
    gpu/
    training/
  benches/
    kernels/
    end_to_end/
```

Recommended initial libraries (versions pinned at implementation time):

- `logos` for lexing; a hand-written recursive-descent/Pratt parser is small enough for M0.
- `miette` or `codespan-reporting` for useful diagnostics with source spans.
- `serde` + `serde_json` for build manifests/checkpoint metadata.
- `blake3` for stable artifact and parameter-identity hashing (with explicit serialization/version rules).
- `ash` for low-level Vulkan access; `rspirv` for direct SPIR-V generation.
- `thiserror` for typed errors; optionally `tracing` for compile/runtime spans.

The **core compiler must compile and test without installing Vulkan**. A `vulkan` Cargo feature gates backend/runtime code; the pure `cpu` backend remains always usable. Neither requires Python or CUDA.

### 7.1 Key API and ownership rules

- Compiler passes take immutable or explicitly mutable owned IR, never shared global mutable graph state.
- The compiler emits a `CompiledArtifact { manifest, kernels, plan }`; the runtime never reparses source. A CPU-backend artifact has no kernels and its plan has no dispatch steps. Each parameter has one or two views (two admit ping-pong updates, § 6.3; M1 emits one), each data input and target one, and the artifact names its loss and prediction views; the schema hash covers parameter paths, dtypes and shapes only, so it is equal across backends and learning rates.
- `CompiledArtifact` is versioned. Untrusted serialized artifacts are validated before dispatch.
- GPU resources are RAII-owned; destruction waits for or tracks in-flight use. Hide unsafe Vulkan calls behind compact, tested abstractions.
- Every buffer view stores dtype, logical element count, allocation range, alignment, and device ownership; the public API prevents mixing incompatible buffer views.
- No raw unvalidated source-level pointer arithmetic or user-injected SPIR-V in M1.

---

## 8. CLI and developer workflow

```sh
modelspec check examples/linear_regression.mspec
modelspec ir examples/linear_regression.mspec --stage tensor
modelspec ir examples/linear_regression.mspec --stage training
modelspec inspect-device --backend vulkan
modelspec compile examples/linear_regression.mspec --backend vulkan --output build/linear/
modelspec train examples/linear_regression.mspec --data examples/regression.msdata --steps 500 --checkpoint run.msckpt
modelspec infer examples/linear_regression.mspec --checkpoint run.msckpt --input examples/query.msdata
modelspec bench examples/linear_regression.mspec --backend vulkan --warmup 100 --iterations 1000
```

M0 CLI implements `check`, `ir`, and `train --backend cpu`. M0 `train` takes `steps`, `seed` and the optimizer from the `train` block, trains on synthetic data (§ 2.3) and prints an `init @<path>: <type> = [...]` line per parameter, a `step <k> loss <v>` line at step 0, every 50 steps and at the last step, and a `param @<path>: <type> = [...]` line per parameter. M1 adds device discovery, Vulkan compilation, GPU training, inference, and checkpoint compatibility. `inspect-device` must report each relevant capability (including whether cooperative matrices are actually supported), not infer it from `RTX 3090` as a product name.

`--emit-spv` (or manifest output from `compile`) allows inspecting generated SPIR-V. Print readable tensor/training/kernel IR with stable IDs to make optimization decisions debuggable.

### 8.1 Errors are part of the language

Examples:

```text
E-TYPE-012: matmul contracting dimension mismatch
  model.mspec:17:18
  lhs: Tensor<f32, [64, 32]>
  rhs: Tensor<f32, [64, 128]>
  expected rhs dimension[0] = 32; got 64
```

```text
E-TARGET-034: no supported f16 storage/compute path on selected Vulkan device
  requested: f16
  available: f32
  suggestion: compile with --compute=f32
```

M0 codes for `train`: E-TARGET-001, the selected backend is not available in this build (refused before the file is read); E-DATA-001, the `train` block does not declare exactly one `input` and one `target`; E-DATA-002, the `target` is not of the model output type; E-DATA-003, a declared `input` or `target` is not read by the loss. M1 adds E-TARGET-010 for `ir --stage kernel` and every Vulkan compile: a tensor of the training step needs more than 65,535 workgroups in each of three dimensions, more than 2^32 invocations in one dispatch (a 32-bit invocation index), or a buffer of more than 4,294,967,295 bytes (Vulkan's `maxStorageBufferRange` is a `uint32_t`), or more bytes than 64 bits count.

Diagnostics distinguish user specification errors, compiler verification failures, missing GPU capabilities, unsupported features, runtime allocation failure, and device loss.

---

## 9. Test and benchmark strategy

**Correctness gates (run before optimizing):**

1. Parser round-trip tests and negative shape/type tests.
2. CPU golden values for each tensor op (including non-square matmuls, transpose and reductions).
3. Finite-difference gradient checks for individual operations and complete linear/MLP graphs; check shared-use adjoint accumulation.
4. Train synthetic linear regression from fixed seeds; verify loss decreases and parameters approach a known solution.
5. Vulkan vs CPU forward, gradients, and post-SGD parameters within documented tolerances over many dimensions, including edges not aligned to workgroup/tile sizes.
6. Run `spirv-val` against each generated module in CI; run validation layers during GPU integration tests.
7. Save/load checkpoint mid-training; verify uninterrupted and resumed **CPU** run equivalence under fixed settings, and GPU run equivalence within defined tolerances.
8. Test invalid/oversized model artifacts, zero dimensions, buffer overflows, kernel bounds, device loss, and out-of-memory errors.

**Performance metrics:** steady-state examples/sec and tokens/sec (where relevant), step latency, per-kernel GPU time, number of dispatches, peak allocated VRAM, host↔device transferred bytes, optimizer overhead, and compile time. Distinguish warm-up/shader-pipeline creation from steady-state training. Use fair baselines with the same shapes, dtype, batch, model semantics, and data transfer accounting.

Initial performance goals are **measured correctness and transparent baselines**, not a blanket speedup target. First get within a useful multiple of comparable GPU libraries for large matmul; then compete by specializing common whole-graph patterns, reducing launch overhead and intermediate memory traffic. One target is gated: ADR-007 sets 10× PyTorch eager throughput on the MNIST benchmark example on the RTX 3090, under its timing rule.

### M1 acceptance criteria (unambiguous)

- [ ] A `.mspec` file defines a linear model or MLP, loss, and SGD without Python.
- [ ] Rust compiler parses, checks, builds typed IR, generates gradients and optimizer step.
- [ ] CPU reference backend trains and validates gradients.
- [ ] Runtime finds RTX 3090 through Vulkan and prints verified device features.
- [ ] Compiler emits valid SPIR-V for f32 elementwise, reduction, transpose, matmul and update kernels.
- [ ] Forward, backward, parameter update execute on GPU without CUDA API or toolkit usage.
- [ ] GPU/CPU results are numerically consistent within specified tolerances.
- [ ] Checkpoint save/load and inference work with the same parameter identities.
- [ ] Built kernels, IR, dispatch graph, and performance timings are inspectable.

---

## 10. Risks and design mitigations

| Risk | Mitigation |
| --- | --- |
| Vulkan backend slower than optimized CUDA libraries | Keep backend abstract; benchmark kernel families separately; optimize tiling, layouts and specialization; do not promise universal superiority. |
| GPU extensions/dtypes vary by driver | Query Vulkan capabilities; have a guaranteed f32 fallback; test actual device, not product assumptions. |
| Poor gradients or numerical instability | Build CPU oracle and finite-difference tests first; use documented f32 reduction and optimizer rules. |
| Autodiff memory explosion | Liveness-aware reuse in M2; explicit checkpoint/recompute planning in M4; memory budget failures are visible. |
| Fusion makes kernels slower | Partition using estimated registers/shared memory and profile results; retain non-fused candidates. |
| Very large specs cause compilation explosion | Bound graph size/dim specialization, intern constants, memoize lowering, expose compilation limits. |
| Driver-specific performance regressions | Cache/re-tune by device and driver identity; keep non-specialized fallback. |
| Unsafe low-level GPU code | Isolate unsafe Rust `ash` calls; static checks, validation layers, bounded address calculation and descriptor ranges. |
| Specification semantics drift with optimizer passes | Version language/IR/operator gradient rules; IR verification and cross-backend differential tests. |

---

## 11. Execution plan / recommended order

**Iteration 1 — language + CPU (approximately 2–6 focused weeks).** Build the lexer/parser, static `Tensor` types and shape checker, typed SSA graph, interpreter, reverse AD, plain SGD, CLI `check/ir/train --backend cpu`. Train linear regression. This is the first full vertical slice.

**Iteration 2 — GPU bring-up (additional weeks to months).** Create a headless Rust `ash` runtime, probe hardware, upload/download f32 buffers, hand-emit a minimal SPIR-V compute kernel via `rspirv`, verify vector add/relu. Add reduction, transpose and naïve matmul; execute generated training step with explicit synchronization. This is the first non-CUDA RTX 3090 trainer.

**Iteration 3 — architecture and optimizations (months).** Elaboration for modules/layers/generics, optimizer IR, liveness and scratch reuse, fusion, tiled matmul, diagnostics, SPIR-V printing, benchmark harness and on-disk kernel cache.

**Iteration 4 — Transformer path (later).** FP16/capability checks, AdamW, softmax/layer norm, embedding lookup, attention, causal and bidirectional masking, GELU, a masked-language-model head for a small BERT-like proof model beside the GPT-like one, mixed precision, numerical loss-handling and cooperative-matrix specialization if the Vulkan driver exposes it.

If the architecture meets its tests, aggressive per-model specialization and autotuning can follow without changing the public specification format.

---

## 12. Open design questions (tracked, not blockers for M0)

- **DSL evolution:** keep the canonical source language and versioned AST stable while introducing modules, compile-time loops, and richer control flow.
- **User-defined gradients:** add `vjp` / `jvp` declarations for primitives with nontrivial differentiation; forbid unsound overrides by default.
- **Low-level escape hatch:** later offer a verified kernel specification language instead of injecting opaque arbitrary shaders into Tensor IR.
- **Dynamic tensor dimensions:** choose recompilation/specialization first, bounded runtime dimensions second.
- **Precision contract:** define permitted reassociation, accumulation types, rounding and overflow for f16/f32 combinations, per op.
- **Data interface:** design a compact stream/batch format that does not smuggle Python-style runtime semantics into the compiler.
- **Autotune storage:** cache performance findings with device/driver versioning; never treat cached timings as universal.
- **Second target:** an optional NVIDIA PTX backend could improve performance using NVIDIA's driver interface, but would no longer satisfy the strict **no CUDA API** requirement for that backend. It must remain optional and never contaminate the source language or core IR.

### Architecture decision records

- **ADR-001 — Rust as implementation language:** accepted. Strong typed ownership model for compiler and low-level runtime; no Python dependency.
- **ADR-002 — Vulkan/SPIR-V as initial GPU backend:** accepted. Independence from CUDA APIs is prioritized over immediate peak throughput.
- **ADR-003 — Explicit typed tensor SSA:** accepted. Required for whole-graph analysis, reverse-mode AD, and optimization.
- **ADR-004 — CPU oracle before GPU performance:** accepted. Preserve independent correctness reference.
- **ADR-005 — Whole training-step compilation:** accepted. Optimizer updates and side-effect ordering are part of compilable IR.
- **ADR-006 — No direct SASS:** accepted for v1. Native instruction compilation remains the vendor driver's responsibility.
- **ADR-007 — 10× PyTorch eager throughput on the MNIST benchmark example:** proposed. On the MNIST benchmark example (784 → 64 ReLU → 10, no bias, `f32`, MSE, SGD lr 0.1, batch 100, 3 epochs, 1,800 steps), ModelSpec on the RTX 3090 trains at least 10× as many images per second as PyTorch eager on the same host and device, with the same test accuracy and final weights within the tolerance of § 9 gate 5, and bit for bit where § 5.4 says so.
  - Baseline: PyTorch `2.14.1+cu130`, eager, on an NVIDIA GeForce RTX 3090 with driver 610.57.04, with `torch.use_deterministic_algorithms(True)`, TF32 off, `set_float32_matmul_precision("highest")`, the host thread pinned to one CPU, and the same data, initialization and shuffle order. Measured on 2026-10-21 (planning store, `verification-report:mnist-benchmark-cuda-baseline`): 519 µs per step, 192,741 images/s, test accuracy 0.9088. The gate is 1,927,410 images/s, 51.9 µs per step. `torch.compile` (933 µs per step) is reported beside it, not gated.
  - Timing rule: the training loop including minibatch gather and one-hot creation, and for ModelSpec command recording, submission and the wait for the last step; excluding loading (the data set upload of § 6.3 included), compilation and pipeline creation, a 20-step warmup, evaluation and IO. Three repeats, the median. ModelSpec runs with `deterministic` on and `fast_math` off (§ 3.3).
  - Relation to ADR-002: the independence from CUDA APIs stays, and the PyTorch side of the benchmark runs outside the repository. ADR-002's priority of independence over throughput stays for every other model; for this example throughput is a gated target. The design is § 6.3, its per-step budget the table there.

---

## References

Primary upstream specifications and crate documentation, for implementation verification:

1. Vulkan SPIR-V environment and validation: https://docs.vulkan.org/spec/latest/appendices/spirvenv.html
2. Vulkan SPIR-V guide: https://docs.vulkan.org/guide/latest/what_is_spirv.html
3. Vulkan cooperative matrix extension: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html
4. Vulkan cooperative-matrix tutorial: https://github.khronos.org/Vulkan-Site/tutorial/latest/Advanced_Vulkan_Compute/09_Specialized_Math/02_cooperative_matrices.html
5. Rust `ash` bindings: https://docs.rs/ash/
6. Rust `rspirv` builder and parser: https://docs.rs/rspirv/
7. Unified SPIR-V specification: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html

**Bottom line:** ModelSpec is a Rust compiler for *mathematical, composable, trainable programs*. Its first credible deliverable is correct end-to-end FP32 training using generated SPIR-V on the RTX 3090 through an owned Rust Vulkan runtime. The compiler architecture makes whole-training-step specialization possible, but numerical correctness and benchmark evidence govern every optimization.
