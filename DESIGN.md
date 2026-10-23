# ModelSpec — Design Specification

**Status:** M0 implemented (iteration 1); M1 onwards proposed  
**Revision:** 0.26 (2026-10-23): the artifact reader refuses a plan of more than 4,096 steps, 65,536 dependencies or 65,536 views with E-ARTIFACT-003 as it parses the manifest, without holding the plan and before the module files and hashes are checked, and the plan verifier's passes keep their state as they go instead of searching every earlier step, so its work on an untrusted artifact is bounded (§ 7.1, § 8.1); 0.25 (2026-10-23): `train --backend vulkan` takes `--checkpoint`, `--checkpoint-every` and `--resume`, a requested checkpoint ending a chunk with one host wait and a parameter read-back, and `infer --backend vulkan [--device <index>]` runs the step plan up to the loss once with one host wait (§ 6, § 6.2, § 8); 0.24 (2026-10-23): `compile` writes an artifact directory, `artifact.json` and one SPIR-V file per kernel, sealed by blake3, and a reader refuses an altered one with E-ARTIFACT-001 to E-ARTIFACT-004 (§ 7.1, § 8, § 8.1); 0.23 (2026-10-23): `train --backend cpu` stops at the first step whose loss or updated parameter is NaN or ±inf, with E-TRAIN-001 and exit 1 (§ 8); 0.22 (2026-10-23): a matmul tile of two rows per invocation reads tiles of up to 16 values of `k` 2 tiles ahead (§ 5.2); 0.21 (2026-10-22): the M1 matmul stages tiles of its operands in workgroup memory, a workgroup of 128 per block of 8 rows by 16 columns, or 16 rows when 8 need at least 256 workgroups, keeping the order of each output element; a matmul's workgroup size is its tile's, the tiles of `k` run in a loop of 4-tile passes, so the size of a module does not grow with `k`, and a tile larger than the device's workgroup memory is refused (§ 3.1, § 5.2); 0.20: one dispatch per computing op except three fusions that keep every rounded value: a `relu` of a `matmul` output is an epilogue of the matmul kernel, which writes the pre-activation too when another op reads it; `p - lr * g` is one `sgd_update` kernel reading the gradient, its product and difference each rounded with `NoContraction`; an `add`, `sub` or `mul` of two compile-time scalars is folded into its reader's immediate when every reader takes one; the fused `lr * g`, a folded scalar and a pre-activation only its epilogue reads have no buffer; the same-bits claim of the fused kernels is verified only on a device without the float-control modes (§ 3.1, § 5.4); 0.19: the `mse` loss sum runs one workgroup of 64 invocations that stage its terms in workgroup memory, in chunks sized by `maxComputeSharedMemorySize`, and invocation 0 adds them in row-major order from the first term; every kernel kind now has workgroup size 64 (§ 3.1, § 5.2); 0.18: `train --backend vulkan --profile` times every dispatch site of every step with timestamp queries read back after the one host wait and prints one `profile dispatch` line per site, the run counters, pipeline creation and compile time and the steady-state time per step after a 20-step warm-up on stderr (§ 6.3, § 8); 0.17: ADR-007 names the working target of 31.4 µs per step, 10× the fastest PyTorch eager CPU figure, and the GPU timing method: a warm-up before the timed window, then the median, the range and the SM clock during the window (§ 12); 0.16: `train --backend vulkan [--device <index>] [--stats]` trains on one device with one host wait per run: the step plan recorded once per step into chained command buffers of up to 256 steps, each step's loss copied into its slot of a device-local buffer, the plan's `Allocate` steps run once at setup, and the parameters and losses read back once; the M1 parts of whole-run recording and read-back; E-TARGET-002 to E-TARGET-004 for `train` (§ 6, § 6.3, § 8, § 8.1); 0.15: the `.msckpt` header byte layout, the 64-byte payload alignment and the payload hash always written, the checkpoint id over the whole manifest, at most 8 dimensions per shape, `train --steps`, `--checkpoint`, `--checkpoint-every` and `--resume`, `infer --backend cpu`, and codes E-CKPT-001 to E-CKPT-004 (§ 6.2, § 8, § 8.1); 0.14: every floating-point reduction runs in an order fixed by the shape alone, with the CPU order one admissible order (§ 3.3); the M1 two-stage reduction, `mse` loss sum, transpose and matmul orders (§ 5.2); the M1 bit-for-bit class, the `mse` loss sum before its division and every matmul output (§ 5.4); the `r` of a reduction and a matmul element and the whole-run L2 bound of § 9 gate 5; 0.13: the GPU tolerance of § 9 gate 5, `GPU_ATOL` and `GPU_RTOL` with `r = |c|` for elementwise, broadcast and SGD kernels, and the reference with subnormal inputs flushed to signed zero on a device that does not report denorm preservation for f32; 0.12: `train --data`, `--init` and `--eval` read dense `f32` tensor files, with codes E-DATA-007 to E-DATA-009 (§ 8, § 8.1); 0.11: the M1 kernel plan, with buffer views, a step plan and a forward plan that end with their one fence instead of a completion point, two-stage reductions, and E-TARGET-010 for a tensor too large to dispatch or bind (§ 3.1, § 7.1, § 8.1); 0.10: `inspect-device`, the required Vulkan version 1.2 and the enabled set of the device, and codes E-TARGET-002 to E-TARGET-004 (§ 6, § 8, § 8.1); 0.9: ADR-007 (proposed) gates 10× PyTorch eager throughput on the MNIST benchmark example on the RTX 3090, with the whole-run schedule that reaches it and the GPU results compared bit for bit (§ 1, § 4, § 5.2, § 5.4, § 6, § 6.1, § 6.3, § 9, § 12); 0.8: the input adjoint, the gradient of a rank-0 target with respect to named inputs in its own program (§ 3.3 pass 4, § 4); 0.7: M3 adds a small BERT-like masked-language-model proof model beside the GPT-like one (§ 1, § 11 iteration 4); 0.6: derivation of parameter, teacher and synthetic-data draws defined (§ 2.3), M0 `train` output and codes (§ 8, § 8.1); 0.5: a train loss calls `model.forward` exactly once (§ 2.1)  
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

M1 schedules the Kernel IR and the Execution DAG together (`schedule::schedule`, printed by `modelspec ir --stage kernel`): one buffer and one view per value and per parameter, except that the SGD update writes its parameter's buffer and view in place rather than a buffer of its own and that the fusions below store no `lr * g` product, no folded scalar and no pre-activation that only its epilogue reads, scalars compiled into kernels as immediates, one dispatch per computing op except for three fusions, and a full barrier on every read-after-write, write-after-read or write-after-write hazard. The fusions keep every rounded value the unfused plan computes. A `relu` whose operand is a `matmul` output is an epilogue of that matmul's kernel (`epilogue Relu`): each invocation applies the same compare and select to the rounded output element before it stores it; when another op reads the pre-activation (the `relu` gradient does), the kernel writes both views (`writes pre-activation`), and otherwise only the activation, with no buffer for the pre-activation. An SGD update `p - lr * g` with a compile-time `lr` is one `sgd_update` kernel that reads `p`, `lr` as an immediate and the gradient view `g`, and computes the product and then the difference, each rounded. An `add`, `sub` or `mul` of two compile-time scalars is folded at compile time into the immediate of its reader, when both operands and the result are finite and not subnormal and every reader takes an immediate; otherwise it stays a kernel, as a `relu` or `relu_grad` of compile-time scalars always does; the folded value is the round-to-nearest-even `f32` result, and a device test checks it against the product the unfused kernel computes on the device. A full reduction is four steps: an `Allocate` of its partials buffer, a stage-1 dispatch, a barrier and a stage-2 dispatch. The artifact carries two plans: the step plan, and a forward plan that computes the prediction into its own view and writes no parameter view; a forward that returns an input or a parameter copies it there (a `Copy` step), and one that returns a literal fills it. Neither plan has a completion point: each has exactly one fence, a `Barrier` with no dependencies that is ordered after every earlier step, and it is the plan's last step; `verify_plan` refuses a fence anywhere else. A runner may record a plan back to back as many times as it chooses, the recordings concatenated, each after the fence of the one before, and places completion points itself; `verify_recorded_back_to_back` checks that concatenation as one plan in which each fence closes a recording and a step with no dependencies after a fence is ordered after it. The runner uploads each batch between recordings, outside the recorded plan. Within one recording, `verify_plan` refuses a read of any parameter after its in-place update, and a dispatch whose workgroups differ from its kernel's; an `Allocate` is run setup, so it refuses one whose buffer an earlier step of the same recording used. No data, loss or prediction view lies in a parameter buffer, and each parameter and data view holds its declared type's element count; the loss view holds one element and the step plan writes it, and both plans write the prediction view. Workgroup sizes are not read from a device profile: 64 for every kernel kind but matmul, and a matmul's is that of its tile (§ 5.2), a function of its shape alone, 128 for every shape today; its workgroup count covers its output in blocks of the tile.

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

Do not silently assume `a+b == b+a` at bit level. In `deterministic` mode every floating-point reduction runs in an order fixed by the shape alone: the workgroup size of a reduction is a compile-time constant of its kernel, never read from a DeviceProfile, so one artifact gives the same bits on every run on one device and driver. The CPU order of § 2.2 is one admissible order; the two-stage GPU reduction of § 5.2 is another, and § 5.4 says which results are compared bit for bit and which within tolerance. `fast_math = true` is a distinct opt-in compilation setting that is recorded in the artifact hash. `deterministic` is a compilation setting too: it fixes reduction order, so it is recorded in the artifact hash, and a run reads it from its artifact.

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
- Reductions (`sum`, `mean`): 2-stage reduction. Stage 1 reduces each block of 64 consecutive elements in a pairwise tree in workgroup memory (at stride 32, 16, … 1, invocation `i` adds value `i + stride` when it exists) and writes one partial per workgroup; stage 2 is one workgroup of 64 in which invocation `i` sums partials `i`, `i + 64`, … in ascending order, then the same tree folds the 64 sums, and `mean` divides by the element count. No padding term is added and there are no atomics, so the order is fixed by the shape alone (§ 3.3).
- `mse` loss sum: one workgroup of 64 invocations stages the terms `(a - b) * (a - b)` in workgroup memory, in chunks of at most `maxComputeSharedMemorySize / 4` terms with a barrier after each staging, and invocation 0 adds them in row-major order from the first term, then the division by the element count.
- Matmul: one invocation per output element, bounds-checked on both output dimensions, over ascending `k`, starting from the `k = 0` product and not from zero; no split-K. The M2 tiling below, done in M1: a workgroup of 128 invocations computes a block of 8 rows by 16 columns (fewer columns, a power of two, when `n` is smaller), or of 16 rows, two per invocation each with its own accumulation, when the block of 8 rows needs at least 256 workgroups: on the RTX 3090 two rows take the `[784, 100] x [100, 64]` matmul of the MNIST example (392 workgroups of 8 rows) from 5.24 to 4.92 µs and are slower at its other four matmuls of at most 52 workgroups (11.4 against 8.6 µs at the forward one, means of 3 profiled runs of 144,000 steps); staging tiles of up to 64 values of `k` of both operands in workgroup memory, `b` transposed with its columns padded to a multiple of 4 plus 4 against bank conflicts; each tile is read into registers 6 tiles ahead and staged in one of two buffers, with one barrier per tile. A tile of two rows per invocation takes tiles of up to 16 values of `k` read 2 tiles ahead instead: on the RTX 3090 they take the `[784, 100] x [100, 64]` matmul from 4.91 to 4.58 µs against tiles of 32 read 6 ahead (medians of 3 profiled runs of 144,000 steps). From the second tile on, the full tiles that read a full tile that many tiles ahead run in a loop of passes of 4 tiles, the reads not yet staged carried from pass to pass in function variables, and the tiles before, after and between are emitted straight, so the size of a module is bounded whatever its `k`. Depth halves until the tile fits the 16 KiB every Vulkan device provides, and a tile larger than the device profile's `maxComputeSharedMemorySize` is refused before any pipeline is created. Values outside the matrices are staged but enter no written output, and the last tile has its own trip count, so the sum of each element is the order above.
- Transpose and scalar broadcast: separate kernels or fuse into consumers; M1 transposes rank-2 tensors as one invocation per output element.
- SGD: `p_out = p_in - lr * gradient`, respecting parameter versioning, using in-place update only after liveness proves it safe.

**M2 optimized kernels**

- Workgroup tiling and shared-memory staging for matmul (done in M1, above); choose tile size to fit shared memory and register pressure. In `deterministic` mode one invocation accumulates each output element over the contracting index in ascending order, tile after tile, and the contracting index is not split across invocations (no split-K), so the order of § 2.2 holds.
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

In M1 the bit-for-bit class holds the `mse` loss sum before its division and every matmul output. M1 emits `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for width 32 on those two kernels exactly when the device profile reports the three features, decorates their `OpFAdd`, `OpFSub` and `OpFMul` with `NoContraction` on every device, and the operator run asserts their bit-equality against the CPU when `negotiate` gives the class `bit-for-bit`. A `relu` epilogue (§ 3.1) runs inside the matmul kernel, under its execution modes; it is a compare and a select, no arithmetic. That a fused kernel leaves the bits of the unfused plan is verified only on a device where the matmul kernel declares none of `DenormPreserve` and `SignedZeroInfNanPreserve`: the RTX 3090 reports `shaderDenormPreserveFloat32` false, so M1 emits no float-control mode there, and the epilogue under those modes is untested. The fused `sgd_update` decorates its `OpFMul` and `OpFSub` with `NoContraction` on every device, so the product is rounded before the difference. Two-stage reductions, the `mse` value after its division and every other M1 kernel are checked within tolerance.

---

## 6. Rust runtime design

The runtime has four owned responsibilities:

**Device:** Vulkan instance/device/compute queue selection, feature negotiation, shader modules, pipeline cache, command pools, and error reporting. Prefer a compute-only headless configuration; no graphics swapchain. The runtime requires Vulkan API version 1.2.0 and refuses a device below it (E-TARGET-004). The enabled set is the required version, no optional device feature, no device extension and one queue from the first compute queue family; a later capability that needs a feature or extension adds it to the set with the story that uses it. Negotiation also fixes the numerics class of § 5.4: `bit-for-bit` when the device reports all three float-control values, `tolerance` otherwise.

**Memory:** device-local tensor arena, staging buffers for host upload/download, allocations obeying Vulkan heap/buffer-range/alignment restrictions, liveness-based scratch-buffer reuse, bounds/overflow checks, and out-of-memory diagnostics. Keep checkpoint and optimizer state in separate logical arenas even if physically suballocated.

**Execution:** execute compiled DAG nodes (uploads, dispatches, barriers, reads); construct descriptor resources, push constants, specialization constants, command buffers, and synchronization. Use appropriate transfer/compute barriers (e.g. synchronization2 when available). No hidden host synchronization between operators or between steps: from M1 a training run on a device waits on the host once (§ 6.3 items 4 and 5). The M1 runner executes the artifact's plans as they are scheduled. At setup it makes one device-local buffer per plan buffer, which runs the step plan's `Allocate` steps once, uploads the initial parameters and the synthetic batch once from one staging buffer, and binds every dispatch of the step plan once in one descriptor pool. Each step records the step plan, a plan barrier as a compute-to-compute memory barrier and a plan copy between transfer barriers, then copies the loss into slot `k` of a device-local buffer of `steps + 1` slots. After the last step it records the steps of the step plan up to the loss, which write no parameter, for the loss of the final parameters into slot `steps`, then copies the parameters and the loss slots to one staging buffer. Up to 256 steps go into one command buffer; the command buffers are submitted to the one queue one after the other without a fence, the last behind the run's only fence, and the host waits on it once and reads everything back. The run counts its host waits, submits, dispatches and the bytes copied up and down; `train --stats` prints them (§ 8). Its results are compared with the CPU run within tolerance, so a training run is in the `tolerance` class of § 5.4 on every device: its losses are `mse` values after their division, and its SGD updates and elementwise kernels are outside the M1 bit-for-bit set of § 5.4.

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

Format version 1 (M1). The header is 56 bytes, every integer little-endian:

| Offset | Width | Field |
|---|---|---|
| 0 | 4 | magic, the bytes `MSCK` |
| 4 | 4 | format version, `u32` `1` |
| 8 | 4 | endian marker, `u32` `0x01020304`, so the bytes `04 03 02 01`; the bytes `01 02 03 04` mark a big-endian file, which is refused |
| 12 | 4 | reserved, zero |
| 16 | 8 | manifest size `m`, `u64`, the bytes of the manifest |
| 24 | 32 | checksum, `blake3` over bytes `0 .. 56 + m` with these 32 bytes counted as zero |

The manifest is compact UTF-8 JSON at byte 56 (`spec/domains/checkpoint.yaml`, `Checkpoint`): `checkpoint_id`, `file_size`, `alignment`, `schema_hash` (the artifact's, `story:m1-kernel-plan`), `step`, `payload_hash`, `optimizer`, `optimizer_slots`, `rng` and `parameters`, each record its `path`, `dtype`, `shape`, `offset` and `length`, in path order. The alignment is 64 bytes: the payload starts at the first multiple of 64 at or after `56 + m`, every tensor offset is a multiple of 64, the gaps are zero bytes, and the file ends with the last tensor. M1 always writes `payload_hash`, `blake3` over the bytes from the payload start to the end of the file. `checkpoint_id` is a version-8 UUID derived by `blake3` from the whole manifest with the nil UUID in its place, optimizer, `rng` and payload hash included, so two runs that reach the same state write the same bytes and two checkpoints that differ anywhere carry different ids. A shape has at most 8 dimensions. M1 SGD has no optimizer slots, `optimizer.schedule` is absent, and `rng` is the seed with counter 0, because an SGD step draws no number.

A reader refuses, before it allocates a tensor and in this order: a bad magic, version, endian marker or reserved field (E-CKPT-001); a file shorter than the header or a manifest size past the end of the file (E-CKPT-002); a checksum that differs (E-CKPT-001); a manifest that is not the JSON above, has no `payload_hash`, a `checkpoint_id` that is not a UUID or a shape of more than 8 dimensions, or breaks an invariant of `Checkpoint` (E-CKPT-001), and one with more records than a file of its size can hold, which is refused before the records are kept so that no single allocation grows past the file size plus 4,096 bytes (E-CKPT-002); a `file_size` other than the file's, an alignment other than 64, or an offset or length that is misaligned, overlaps the header, the manifest or another tensor, ends past the file, or is not the shape's element count times the dtype size, a file that does not end at its last tensor, and a byte between the manifest and the first tensor or between tensors that is not zero (E-CKPT-002); and a payload hash that differs (E-CKPT-004). A checkpoint loads into a compiled file only with the same parameter paths, dtypes and shapes and the same schema hash, and a resumed run also needs the same optimizer; the first difference is refused with E-CKPT-003.

### 6.3 Whole-run execution (M2)

M2 executes a training run, or a requested number of its steps, with no host round-trip per operator or per step. M1 already records a run on synthetic data that way (§ 6, Execution): items 4 and 5 say what holds from M1 and what M2 adds. M2 changes the schedule only: every kernel computes the ops of the verified training IR (§ 0 invariants 1 and 5), and § 5.4 says which results equal the CPU oracle bit for bit.

1. **Resident state.** The data set, its labels, the permutation of every epoch the run covers, two versions of each parameter and one loss slot per reported step are uploaded once before the first step and stay in device-local memory; the allocator checks them against the heap budget (§ 6, Memory). For the MNIST benchmark example that is 188,160,000 bytes of images and 720,000 bytes of permutation. The upload is loading, outside the timed region of ADR-007.
2. **Gather on the device.** Step `s` reads its minibatch rows through `permutation[s * B + r]` in the operand load of its first matmul (§ 5.2), and builds its one-hot targets from the labels the same way. The permutation is the shuffle order of the data binding, computed on the host. The gather is a runtime data binding, not a Tensor IR op (§ 2.3: a model does no I/O).
3. **Parameter versions.** Step `s` reads version `s mod 2` of each parameter and writes version `(s + 1) mod 2` (§ 3.1). Two buffers replace an in-place update because the kernel that updates `W2` runs beside the kernel that still reads the old `W2`.
4. **Recording.** The host records the steps in chunks, one command buffer per chunk, with a compute-to-compute memory barrier between dependent dispatches. It records chunk `n + 1` while the device executes chunk `n` and submits it to the same queue. The host waits only at the end of the run, at a checkpoint (§ 6.2) and at a requested read-back; a checkpoint ends a chunk. Steps are not merged into one dispatch behind a device-wide barrier, because Vulkan does not guarantee that the workgroups of one dispatch run at the same time. From M1: chunks of 256 steps, every command buffer submitted without a fence but the last of each segment, a segment ending at each step `--checkpoint-every` requests and at the end, and one host wait per segment; an M1 run has no other requested read-back on the device. M2 adds the step index in a push constant, which the gather of item 2 reads.
5. **Read-back.** Parameters are read back at the end and at each checkpoint. On a reported step (§ 8) the loss is written to its slot and read back with the parameters; the `step <k> loss <v>` lines keep their content and order and are printed after that read-back. Per-kernel timing uses timestamp queries on sampled steps (§ 0 invariant 7). From M1: the parameters and every step's loss, `steps + 1` slots, are read back once at the end, the parameters also at each checkpoint, and `--serve` streams the `StepCompleted` events after the last read-back. M1 `train --profile` writes the timestamp queries on every step and reads them back after the last wait (§ 8). M2 writes only the reported steps' losses, and samples the timestamp queries.
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
- `CompiledArtifact` is versioned. Untrusted serialized artifacts are validated before dispatch. On disk an artifact is a directory: `artifact.json` (format version 1) and, for Vulkan, `kernels/<kernel_module_id>.spv`, one per kernel, its words little-endian; `compile` removes the `artifact.json` and `kernels/` of an earlier artifact in the output directory before it writes, so the directory holds exactly the new artifact. The manifest holds the format version, `artifact_id`, `backend`, `artifact_hash`, the source, IR and schema hashes, `fast_math`, `deterministic`, `compiler_version`, the parameters, data, kernels (module id, entry point, cache key, byte length, blake3 and the Kernel IR, an `f32` immediate by its bits), the step and forward plans, the buffer sizes and the loss and prediction views. `artifact_hash` is blake3 over the pretty-printed JSON of the manifest with `artifact_hash` and `artifact_id` empty, followed by each module's bytes in id order; `artifact_id` is its first 16 bytes as a version-8 UUID. The hash detects an artifact changed after it was written; it does not authenticate one. `modelspec_core::artifact::read` checks, in order: the manifest a regular file and the format version (E-ARTIFACT-001); each of the step and forward plans has at most 4,096 steps, 65,536 dependencies and 65,536 views (reads and writes summed over its steps), counted as the manifest is parsed so that a plan past them is refused without being held, sealed or not (E-ARTIFACT-003); every module file present, a regular file and no longer than its entry, and no entry in `kernels/` that no kernel names, before any module is hashed, no read going past an entry's byte length (E-ARTIFACT-004); each module's blake3, then `artifact_hash` and `artifact_id` (E-ARTIFACT-002); then every byte size fits in 64 bits, the invariants of `spec/domains/artifact.yaml` (no buffer over 4294967295 bytes among them), every Dispatch names a module the artifact owns, every module is SPIR-V (whole words, a 5-word header, the magic number 0x07230203), and the plan verifier (E-ARTIFACT-003). Two compiles of one file on one device profile write byte-equal directories.
- The plan verifier's work is bounded by the plan, so the reader's limits bound it on an untrusted artifact. Each pass keeps what it needs of the steps before the current one (the buffers its recording used or allocated, the last stage 1 per buffer, the readers of each parameter, the first update per buffer in the recording, the Dispatch and Copy steps that touch each buffer), and the steps each step is ordered after are bit sets. Its time and memory grow with the steps times the steps, dependencies and views, over 64: at the limits the two step-by-step bit sets take 4 MiB, and the six plans of 4,096 steps in the shapes that load a pass most, an unsynced pair of Copy steps at the view limit among them, verified in 0.015 s to 0.162 s each in a debug build (three runs, 2026-10-23). In those runs the reader refused a sealed 64 MiB manifest of 33.5 million dependencies in 2.4 s to 4.5 s, at most 71 MiB resident. The other arrays of the manifest have no limit yet: a sealed 64 MiB manifest that fills a parameter or data shape or an array of a kernel's cache key took the reader longer than 10 s or more than 1 GiB, and one that fills the buffers 8.3 s and 518 MiB. The limits are 186 times the largest plan an example compiles to, the 22-step MNIST step plan. The compiler does not refuse a plan past them yet; `compile` would write it and its own read-back refuse it with E-ARTIFACT-003.
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

M0 CLI implements `check`, `ir`, and `train --backend cpu`. M0 `train` takes `steps`, `seed` and the optimizer from the `train` block, trains on synthetic data (§ 2.3) and prints an `init @<path>: <type> = [...]` line per parameter, a `step <k> loss <v>` line at step 0, every 50 steps and at the last step, and a `param @<path>: <type> = [...]` line per parameter. `train --data <dir>` trains on dense `f32` tensors instead (`spec/domains/data.yaml`, `DenseDataDirectory`): the `input` and the `target` declaration `name: Tensor<f32, [B, d1, …]>` each read `<dir>/<name>.f32`, row-major little-endian `f32`, one sample per `d1 × …` values, `N` samples per file. Step `k` feeds the samples `order[k·B + i]` of `<dir>/order.u32` (little-endian `u32` sample indices) when it exists and `(k·B + i) mod N` otherwise, for `i` in `0..B`; the loss after the last update reads step `steps`, positions past the end of the order wrapping. `--init <dir>` starts each parameter with a file `<dir>/<p>.f32`, `p` its canonical path without the model name, from that file; the others keep their initializer, and a file that names no parameter is ignored. `--eval <dir>` runs the forward program after the last step over every sample of `<dir>` in batches of `B` and prints a last line `eval loss=<mean of the batch losses> accuracy=<fraction>`, a sample counting as correct when the first maximum of its output row is at the first maximum of its target row and never when its output row holds a NaN. Every file is read before the first step. Without these flags `train` is as above. M1 adds device discovery, Vulkan compilation, GPU training, inference, and checkpoint compatibility; M1 inference and checkpoints run on the CPU and on the device, a checkpoint written by either backend loading into both. `inspect-device` must report each relevant capability (including whether cooperative matrices are actually supported), not infer it from `RTX 3090` as a product name.

`inspect-device --backend vulkan [--device <index>] [--format text|json]` prints the device profile (`spec/domains/device.yaml`) of every Vulkan physical device, or of the one `--device` names, as the driver reports it; a value the device's API version has no query for (below Vulkan 1.2: the float controls and f16 arithmetic, below 1.1 also the UUID, subgroup size and 16-bit storage) is `null`, never a default. Text prints one `<field>: <value>` line per field and a `negotiation: ok` or `negotiation: refused (<code>)` line; json prints the profile alone. `MODELSPEC_VULKAN_LIBRARY` names the Vulkan loader to open instead of the system's `libvulkan`. A build without the `vulkan` feature refuses with E-TARGET-001.

`train <file> --backend vulkan [--device <index>] [--stats] [--profile]` compiles the file for Vulkan and trains it on device 0, or on the device `--device` names in the loader's order, from the same seed, synthetic data and `steps` (or `--steps`) as the CPU run, with one host wait for the whole run and one more per checkpoint `--checkpoint-every` requests (§ 6, Execution). It prints the lines a CPU run prints, in the same order, each value the device's, within the tolerance of § 9 gate 5 of the CPU run's. `--stats` prints the run's counters on stderr after it, one `<name>: <value>` line each: `steps`, `host waits`, `submits`, `dispatches`, `dispatches per step`, `bytes uploaded`, `bytes downloaded`, `numerics class` and `us per step`, the time from the first submit to the end of the one wait over the steps. `--profile` writes a timestamp before and after each dispatch of every step into one query pool per command buffer and reads them back after the run's one wait, adding no host wait; on stderr after the `--stats` lines, or after the run, it prints one line per dispatch site, that is per `Dispatch` step of a plan, `profile dispatch <plan> <position> <kernel_module_id> <entry> calls <n> total_us <t> mean_us <m>`, two sites that name one module being two lines. `<plan>` is `step` for the step plan, `calls` the number of steps, and `final` for the final loss evaluation, the steps of the step plan up to the loss recorded once after the last step, `<position>` the index of the step in the step plan. Then come `profile dispatches`, `profile submits`, `profile host_waits`, `profile bytes_uploaded`, `profile bytes_downloaded`, `profile pipeline_creation_us`, `profile compile_us` (emitting the kernels and compiling the file) and `profile steady_us_per_step`, the GPU time from the first dispatch of step 20 to the first dispatch of the final loss evaluation over the steps after the 20-step warm-up, `unavailable` for a run of at most 20 steps. On a device without timestamp queries the dispatch lines are one `profile kernel timings unavailable: no timestamp queries` line and the counter lines are printed. Stdout is unchanged. `--data`, `--init` and `--eval` run on the CPU only and `--device`, `--stats` and `--profile` on Vulkan only; naming one with the other backend is a usage error. `--serve` streams the run as on the CPU, its `StepCompleted` events after the one read-back. The file is compiled before the loader is opened, so a file that does not compile reports as on the CPU. A build without the `vulkan` feature refuses `--backend vulkan` with E-TARGET-001 before the file is read.

M1 checkpoints on both backends (§ 6.2). `train --steps <n>` overrides the `train` block's `steps`. `train --checkpoint <path>` writes the state after the last step to `<path>`, and `--checkpoint-every <n>` also writes `<stem>-<k>.msckpt` beside it, `<stem>` the file name of `<path>` without its extension, at every step `k` after the start that is a multiple of `n` and before the end. `train --resume <path>` starts from the parameters and step `s` of a checkpoint instead of the initializers and step 0, refusing one that does not match the compiled file or its optimizer (E-CKPT-003); it runs to step `s + steps`, prints its `init` lines at step `s`, its `step` lines at step `s`, at every multiple of 50 and at the last step, feeds dense data from step `s`, so `order.u32` must hold `(s + steps) · B` indices, and cannot be combined with `--init`. A resumed CPU run writes the same checkpoint bytes and prints the same `step` lines from step `s` as a run that never stopped (§ 9 gate 7). `infer <file> --backend cpu --checkpoint <path>` runs the forward program with the checkpoint's parameters, refused with E-CKPT-003 unless they match the compiled file, on the synthetic batch of the `train` block and prints `prediction: <type> = [...]` and `loss <v>`; it takes no `--input` in M1 (§ 12 "Data interface"). On Vulkan a checkpoint ends a segment of the run: the host waits once for it, reads the parameters back and writes the file, then records the next segment from the parameters still on the device, so a run with `--checkpoint-every` waits once per such checkpoint and once at the end, and a resumed device run passes against the run that never stopped within the whole-run L2 bound of § 9 gate 5 for its final parameters and the tolerance of § 9 gate 5 for each loss. `infer <file> --backend vulkan [--device <index>] --checkpoint <path>` reads the checkpoint before it opens the loader, so a checkpoint the CPU refuses is refused with the same E-CKPT code; it then runs the steps of the step plan up to the loss once on the device, with one host wait, and prints the same two lines, each value within the tolerance of § 9 gate 5 of `--backend cpu`'s. A build without the `vulkan` feature refuses `infer --backend vulkan` with E-TARGET-001 before the file is read.

`compile <file> --backend cpu|vulkan --output <dir> [--device-profile <json>]` writes the artifact directory of § 7.1, reads it back through the artifact reader and prints `artifact <artifact_id> <artifact_hash> kernels=<n> <dir>`. A Vulkan compile builds its modules for the `DeviceProfile` JSON file `--device-profile` names (the form `inspect-device --format json` prints) or, without it, for device 0 of the loader, so a missing loader exits 1 with E-TARGET-002 and a compile with a profile opens no loader. The profile is negotiated before anything is written: a profile below Vulkan 1.2 exits 1 with E-TARGET-004 and the output directory is not created, and a profile file that cannot be read or breaks an invariant of `spec/domains/device.yaml` exits 1 naming the file and the invariant. A Cpu artifact has no `kernels/` and its compile opens no loader; `--device-profile` with `--backend cpu` is a usage error. The Cpu and Vulkan artifacts of one file have equal schema hashes and parameters. `compile` replaces the `--emit-spv` of earlier drafts: `spirv-dis kernels/<id>.spv` shows a module. Print readable tensor/training/kernel IR with stable IDs to make optimization decisions debuggable.

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

M0 codes for `train`: E-TARGET-001, the selected backend is not available in this build (refused before the file is read); E-DATA-001, the `train` block does not declare exactly one `input` and one `target`; E-DATA-002, the `target` is not of the model output type; E-DATA-003, a declared `input` or `target` is not read by the loss. Codes of `train --data`, `--init` and `--eval` (§ 8), each before the first step: E-DATA-007, a data, order or init file is missing, unreadable, or not a whole number of samples (for `--init`, not the parameter's size), a data or init file holds a NaN or an infinity, the data files differ in sample count, the `input` and `target` do not share a leading dimension `B`, or the `--init` directory cannot be read or holds no file of any parameter; E-DATA-008, `order.u32` holds fewer than `steps · B` indices or an index `≥ N`; E-DATA-009, an `--eval` sample count that is not a multiple of `B`. E-DATA-001 to E-DATA-003 apply to dense data as to synthetic data. M1 adds E-TARGET-010 for `ir --stage kernel` and every Vulkan compile: a tensor of the training step needs more than 65,535 workgroups in each of three dimensions, more than 2^32 invocations in one dispatch (a 32-bit invocation index), or a buffer of more than 4,294,967,295 bytes (Vulkan's `maxStorageBufferRange` is a `uint32_t`), or more bytes than 64 bits count.

Code of `train --backend cpu` during the run: E-TRAIN-001, the loss of a step, or a parameter after the update of a step, is NaN or ±inf. The run stops at the first such step, the loss checked before the parameters: stdout keeps its `init` lines and the `step` lines printed so far and ends with the `step <k> loss <v>` line of the failing step, with no `param` lines; stderr has the diagnostic naming the step and either the loss or each non-finite parameter by canonical path; the command exits 1 and `--serve` ends the stream with that step's `StepCompleted` (its loss `null` when the loss is the non-finite value), then `RunFailed`. A run that stays finite prints what it printed before. The Vulkan backend does not check yet.

M1 codes for `inspect-device` and for `train --backend vulkan`, after the file compiles and before the first step: E-TARGET-002, no Vulkan loader could be opened; E-TARGET-003, no Vulkan instance or physical device was found, `--device` names none of them, or the device has no compute queue family; E-TARGET-004, the device is below the required Vulkan API version 1.2.0 (§ 6).

M1 codes for checkpoints of `train --resume`, `train --checkpoint` and `infer` (§ 6.2), each before the first step: E-CKPT-001, not a readable checkpoint (it cannot be opened, or its magic, version, endian marker, reserved field, checksum or manifest is wrong); E-CKPT-002, an offset, length or size outside the file or misaligned, a non-zero byte between tensors, bytes after the last tensor, or more records than the file can hold; E-CKPT-003, the checkpoint does not match the compiled file (a parameter path, dtype or shape, the schema hash, or for `--resume` the optimizer); E-CKPT-004, the payload hash differs.

M1 codes of the artifact reader (§ 7.1), which `compile` runs on what it wrote, in the order the reader checks them: E-ARTIFACT-001, `artifact.json` is missing, unreadable, larger than 64 MiB, not a regular file or not a manifest, or its format version is not 1; E-ARTIFACT-003 for a plan of more than 4,096 steps, 65,536 dependencies or 65,536 views, refused as the manifest is parsed, whether the artifact is sealed or not; E-ARTIFACT-004, a module file the manifest names is missing, not a regular file, unreadable or longer than its entry, or `kernels/` holds an entry no kernel names; E-ARTIFACT-002, a module's bytes differ from its blake3, or `artifact_hash` or `artifact_id` is not the one the content gives; E-ARTIFACT-003, a sealed artifact breaks another invariant: a byte size past 64 bits, a buffer over 4294967295 bytes, a module that is not SPIR-V, a Dispatch naming a module the artifact does not own, a dependency on a step not before it, a view outside its buffer, or another rule of `spec/domains/artifact.yaml` or the plan verifier.

Diagnostics distinguish user specification errors, compiler verification failures, missing GPU capabilities, unsupported features, runtime allocation failure, and device loss.

---

## 9. Test and benchmark strategy

**Correctness gates (run before optimizing):**

1. Parser round-trip tests and negative shape/type tests.
2. CPU golden values for each tensor op (including non-square matmuls, transpose and reductions).
3. Finite-difference gradient checks for individual operations and complete linear/MLP graphs; check shared-use adjoint accumulation.
4. Train synthetic linear regression from fixed seeds; verify loss decreases and parameters approach a known solution.
5. Vulkan vs CPU forward, gradients, and post-SGD parameters within documented tolerances over many dimensions, including edges not aligned to workgroup/tile sizes. A device element `g` passes against the CPU interpreter's `c` when `|g - c| <= GPU_ATOL + GPU_RTOL * r`, with `GPU_ATOL = 1e-5` and `GPU_RTOL = 1e-4`; elementwise, broadcast, SGD and transpose kernels use `r = |c|`; for a reduction, loss-sum or matmul element `r` is the sum of the absolute values of its terms (the elements of a `sum`, the `K` products of a matmul element), divided by the element count for `mean` and for `mse` over its squared differences (story:m1-vulkan-reduce-matmul). After a run every parameter passes `||g - c||_2 <= 1e-4 * ||c||_2`, and a bit-for-bit result also passes the tolerance (§ 5.4 item 3). The tests name the tolerance per operator, and each per-kernel case (one kernel per command buffer, not the chained add, relu and SGD case) also checks that the 16 guard words after its output stay unchanged; an SGD update's gradient carries guard words too, so a read past its end changes the parameter's guard words. On a device that does not report denorm preservation for f32 (`shaderDenormPreserveFloat32` false or not reported) the reference `c` is the CPU interpreter run on the inputs with every subnormal flushed to a zero of its sign.
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
  - Working target: 31.4 µs per step. PyTorch eager on the same host's CPU is fastest at 8 threads, 314 µs per step median (planning store, `verification-report:pytorch-cpu-multithread-2026-10-22`, measured on 2026-10-22 under load, so an upper bound on the quiet time), faster than PyTorch eager on CUDA. 10× over the fastest PyTorch eager figure on this host is 31.4 µs per step, below the 51.9 µs gate above; the work aims at 31.4 and the gate stays 51.9.
  - GPU timing method: the RTX 3090 idles at P8 and a run under one second stays at P3 (780 MHz of 2130 MHz), so a figure from a cold device measures the clock floor, not the schedule. Before the timed window the device runs the same program for at least 5 s; the timed window is at least 5 runs of at least 5 s each, back to back; `nvidia-smi --query-gpu=timestamp,clocks.sm,pstate` samples every 200 ms throughout. The report gives the median and the range of the per-run steady-state µs per step and the share of samples per performance state with its mean SM clock during the window. Clocks are not locked. No other compute process is on the device.
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
