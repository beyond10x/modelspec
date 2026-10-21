# ModelSpec compilation targets

**Status:** proposal, design only; nothing in this document is implemented.
**Date:** 2026-10-21. **Base:** `main` at `a7501f0` (DESIGN.md revision 0.6).
**Authority:** DESIGN.md stays the authority. Section 11 lists the changes this document proposes
to it and section 12 the ESS changes; none is applied.

**Question.** Which targets can one `.mspec` training step be compiled to? Does training on a GPU
inside a browser make sense, and what about inference there? What else exists? And, as an
experiment: can the compiler emit highly optimised machine code directly?

**Evidence marks.** Every specific is marked.

| Mark | Meaning |
| --- | --- |
| **[V]** | Verified: read in the cited source on 2026-10-21, observed in this repository at `a7501f0`, or observed on the reference machine (§ 1.4). |
| **[I]** | Inferred: reasoning or arithmetic from verified facts; not measured, not read in a source. |
| **[H]** | Hypothesis: a claim this document defines an experiment to test. |

Source keys such as `[W3]` refer to the source register in section 13; the raw list with
quotations is in `targets-research-notes.md`.

## 0. Summary

| Question | Answer | § |
| --- | --- | --- |
| Browser GPU training | Makes sense for small models (training state up to about 1.5 GiB), teaching, private fine-tuning and as a second GPU check. Not for throughput: the best published end-to-end run reaches about 1.4 % of its GPU's `f32` peak. Never bit-exact: WGSL permits reassociation, fusion, flushed denormals and finite-math assumptions. | 3 |
| Browser inference | Yes. Mature stacks reach up to 80 % of native speed on Apple silicon, much less on NVIDIA. For ModelSpec it is the forward half of the same web bundle. | 3.6 |
| Architecture | Two branch points: graph exports after the Training IR, code targets after a target-neutral Kernel IR and a Scheduled IR, each with a reference interpreter. A pure `Target` trait emits; executors own devices. Every target declares a `NumericContract`; the oracle runs any deterministic contract bit for bit; tolerances between contracts are computed from rounding-error bounds. | 4, 5 |
| Order | Kernel IR and Scheduled IR → Vulkan (M1), with the x86-64 experiment beside it → safetensors → browser inference → browser training → ONNX. Everything else is research with a named trigger. | 6 |
| M1 | Same scope. It builds the Kernel IR first, emits SPIR-V from a device-free crate, and decorates arithmetic `NoContraction` in `deterministic` mode. | 7 |
| Machine-code experiment | x86-64 AVX2, a `6 × 16` register-blocked kernel JIT-encoded with `iced-x86`. Success: exact mode bit-identical to the interpreter (S1); at least 80 % of the 16 FLOP per cycle exact ceiling (S2) and of the 32 FLOP per cycle FMA ceiling, and at least 90 % of OpenBLAS (S3), on one core at 1024³ to 4096³; at `64 × 32 × 64`, `relu(matmul)` plus the exact sum no slower than `matrixmultiply`'s matmul alone (S4). | 8 |
| Open decisions | D1 GPU stack sharing, D2 browser execution form, D3 emission mechanism, D4 CPU default mode, D5 experiment timing. | 10 |

## 1. Starting point

### 1.1 What every target consumes [V]

A target receives a verified, statically shaped `f32` program. The Tensor IR has 14 operations
(`crates/modelspec-core/src/ir/tensor.rs`, `TensorOp`): `Input`, `ParamRead`, `Scalar`, `Add`,
`Sub`, `Mul`, `MatMul`, `Relu`, `Transpose`, `ReduceSum`, `ReduceMean`, `Mse`, `ReluGrad` and
`BroadcastScalar`. A verified program has no symbolic dimension. The Training IR
(`ir/training.rs`) tags nodes `Forward`, `Loss`, `Gradient` and `Optimizer` and orders parameter
reads and writes with effect tokens; `compile_training` (`compile.rs`) returns the training step,
the lowered forward program and the train configuration. SGD is expressed with the same
operations (`Sub`, `Mul` by a scalar), so a target that implements the 14 operations runs every
program M0 can compile.

That set is small, and every backend decision below is sized to it. M3 adds softmax, layer norm,
embeddings, attention and AdamW (DESIGN.md § 1); the architecture has to absorb them, but no
target has to be designed for them now.

### 1.2 The oracle [V]

The CPU interpreter (`crates/modelspec-runtime/src/cpu/interpreter.rs`) is generic over the
element type, so one program runs in `f32` and in `f64`. Its arithmetic is the reference every
target is compared with:

- A sum folds sequentially from its first term (`sum`, line 338), not from `+0`. `matmul` computes
  each element as that sum of `a[i,k] * b[k,j]` over ascending `k` (line 320). Nothing is fused.
- `relu(x)` is `if x > 0 { x } else { 0 }` (`relu_where`, line 289). So `relu(-0.0)` is `+0.0` and
  `relu(NaN)` is `+0.0`. `ReluGrad(a, g)` is `if a > 0 { g } else { 0 }`.
- `mean` and `mse` divide the sum by the element count converted to the element type.

The tests that hold it: `tests/cpu_golden.rs` (bit-exact `f32` and `f64` values per operation),
`tests/gradients.rs` (central differences in `f64`, `H = 1e-5`, `TOL_F64 = 1e-6`,
`TOL_F32 = 1e-3`, `KINK_MARGIN = 1e-2`) and the IR printouts under
`crates/modelspec-cli/tests/golden/`.

Rust's arithmetic on `f32` produces "results that exactly match IEEE 754-2008", which "precludes
many transformations, such as turning a*b + c into FMA operations" [R1]. The interpreter's results are therefore a
property of the program, not of the compiler that built the interpreter, except for NaN payloads,
which IEEE leaves unspecified ("IEEE makes almost no guarantees about the sign and payload bits of
the NaN" [R1]).

### 1.3 Constraints every target inherits [V]

| Constraint | Where it is stated |
| --- | --- |
| The core compiles and tests without Vulkan; `no_vulkan.rs` refuses `ash`, `vulkano`, `wgpu`, `gpu-allocator` and `vk-mem` in its dependency graph. | DESIGN.md § 7; `crates/modelspec-core/tests/no_vulkan.rs` |
| No Python and no CUDA API or toolkit. An optional PTX backend may exist but must not touch the language or core IR. | DESIGN.md § 0, § 12 |
| No direct SASS (ADR-006); no user-authored GPU assembly; no user-injected SPIR-V in M1. | DESIGN.md § 1, § 7.1, ADR-006 |
| No reassociation of floating-point reductions in `deterministic` mode; `fast_math` and `deterministic` are recorded in the artifact hash. | DESIGN.md § 3.3 |
| GPU results are checked against documented tolerances, not bit for bit. | DESIGN.md § 4 |
| Every program that runs is Rust; browser code is Vue 3 with TypeScript, built with Vite and pnpm; types shared with Rust are generated from Rust and drift-checked. | organisation rule |
| A new data contract gets its ESS type before code. | AGENTS.md, organisation rule |
| The artifact has `Backend` = `Cpu` or `Vulkan`, a kernel module carries `spirv: Bytes`, and a Cpu artifact has no Dispatch steps. | `spec/domains/artifact.yaml` |
| `device_profile_id` is "Present exactly when the artifact's backend is Vulkan"; a checkpoint is "loadable by the CPU backend whichever backend wrote it". | `spec/domains/training.yaml:76`, `spec/domains/checkpoint.yaml:74` |

### 1.4 Reference machine [V]

Read-only inventory on 2026-10-21 (`lscpu`, `/proc/cpuinfo`, `vulkaninfo`; no compute ran on the
GPU).

| Item | Value |
| --- | --- |
| CPU | Intel Core i9-10900K, family 6 model 165, 10 cores, 20 threads, maximum 5.3 GHz |
| Vector ISA | `avx`, `avx2`, `fma`, `f16c`; no AVX-512 flag |
| Caches | L1d 32 KiB and L2 256 KiB per core, L3 20 MiB shared |
| Memory | 62 GB |
| Performance counters | `perf_event_paranoid = 2` (own-process user-space counting allowed), `perf` installed |
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04, Vulkan API 1.4.341 |
| GPU limits | `maxComputeSharedMemorySize` 49152, `maxComputeWorkGroupInvocations` 1024, subgroup size 32, `maxStorageBufferRange` 4294967295 |
| GPU float controls | `shaderDenormPreserveFloat32` false, `shaderSignedZeroInfNanPreserveFloat32` true, `shaderRoundingModeRTEFloat32` true |
| GPU extensions | `VK_KHR_cooperative_matrix` (feature on), `VK_NV_cooperative_matrix2`, `VK_EXT_shader_atomic_float` (`shaderBufferFloat32AtomicAdd` true), `VK_KHR_shader_float_controls2`, `shaderFloat16` true |
| Software Vulkan | no lavapipe ICD installed (`/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` absent) |
| Toolchain | rustc 1.99.0; targets `wasm32-unknown-unknown`, `wasm32-wasip1`, `aarch64-apple-darwin`, `x86_64-unknown-linux-{gnu,musl}`; LLVM 22.1.8; wasmtime 48.0.1 |
| Libraries | OpenBLAS 0.3.34, oneDNN 3.11.3, reference BLAS 3.12.1, CLBlast 1.7.0 |
| Browsers | Chrome 150.0.7871.46, Firefox 158.0.1 |

The float-controls row matters for conformance: this GPU does not promise to preserve `f32`
denormals, so a Vulkan result may flush a value the CPU oracle keeps (§ 5.1).

## 2. What exists

Read on 2026-10-21 unless marked; keys refer to § 13.

### 2.1 The browser: WebGPU, WGSL, WebAssembly, WebNN

| Topic | Finding | Source |
| --- | --- | --- |
| Chrome | Ships WebGPU since 113 on macOS, Windows x86-64 and ChromeOS, since 121 on Android 12+. On Linux: Intel Gen12+ since 144, NVIDIA with driver 535.183.01+ on Wayland since 147, other GPUs behind a flag | [W1, W2, W3] |
| Firefox | 141 on Windows; macOS 26 on Apple silicon since 145, and per the wiki all macOS versions since 147 (MDN's data lists no Intel Mac); Linux in Nightly only | [W1, W4] |
| Safari | 26, macOS and iOS | [W1, W4] |
| Reference machine | Chrome 150 and NVIDIA driver 610.57.04 meet the Linux NVIDIA condition except Wayland, which was not inventoried: I don't know whether this session runs Wayland | [V, § 1.4] |
| Default limits | `maxStorageBufferBindingSize` 128 MiB, `maxBufferSize` 256 MiB, workgroup storage 16 384 B, 256 invocations per workgroup | [W5] |
| Adapter limits in the field | `maxStorageBufferBindingSize` ≥ 1.5 GiB on 79 % of reports, ≥ 2 GiB (the bucket is 2³¹ − 1) on 16 %; `maxBufferSize` ≥ 2 GiB on 79 %; the page carries no date | [W8] |
| Optional features | `shader-f16` since Chrome 120, on Linux NVIDIA since 155–156; subgroups since 134; timestamp queries, quantised to 100 µs | [W9, W11, W10, W12] |
| Atomics | `atomic<T>`: "T must be either u32 or i32"; `f32` atomics are the open issue gpuweb#4894, whose one measurement (16 keys, 4 M rows, Apple GPU) puts an `f32` compare-and-swap sum at 366 ms against 2.9 ms for a native `u32` count, not like for like; at 1 024 keys 4.4 against 2.5 ms | [W6f, W7] |
| Floating point | "An implementation may reassociate operations"; may fuse when "at least as accurate"; inputs and outputs of the operations in the accuracy table "may be flushed to zero", while "Other operations are required to preserve subnormal numbers"; `+ − ×` correctly rounded, `f32` division 2.5 ULP for `|y|` in `[2⁻¹²⁶, 2¹²⁶]`; "Implementations may assume that overflow, infinities, and NaNs are not present" | [W6a–W6e] |
| Ordering between dispatches | Each dispatch of a compute pass is its own usage scope; neither specification was found to say in so many words that one dispatch's storage writes are visible to the next | [W6g; V: not found] |
| Lifecycle | The user agent may lose a device whenever it "needs to revoke access"; hidden tabs run chained timers once per second, and once per minute after five minutes hidden under further conditions; animation frames wait for visibility; Memory Saver "deactivates" unused tabs, which reload when opened | [W5, W13, W14] |
| WebAssembly | `wasm32` memory up to 4 GiB; Memory64 in Chrome 133 and Firefox 134, Safari only in Technology Preview; the JS API caps it at 16 GB, and it measured 10 % to over 100 % slower; threads need cross-origin isolation; relaxed SIMD's `relaxed_madd` rounds once or twice, by hardware | [W15, W16, W17, W18] |
| WebNN | Candidate Recommendation Draft of an inference API; the working group's charter, not the specification, puts training out of scope; origin trial from Chrome 147 per chromestatus as last updated 2026-04-08, not shipped | [W19, W20] |
| Native WebGPU | `wgpu` 30.0.1: web and native `SHADER_F16`, `TIMESTAMP_QUERY`; native only `SUBGROUP`, `SHADER_FLOAT32_ATOMIC`, `EXPERIMENTAL_COOPERATIVE_MATRIX` ("currently only supports 8x8 f32 matrices"); `naga` writes SPIR-V, Metal and HLSL as primary backends and WGSL and GLSL as secondary; Chrome compiles WGSL with Dawn and Tint, not `naga` | [W21, W22, W23] |

What follows for ModelSpec [I]:

- The WGSL emitter writes `select(0.0, x, x > 0.0)`, never `max(x, 0.0)`: under the finite-math
  permission NaN has no defined WGSL result, and `select` at least keeps the code literal to the
  oracle's `relu_where`.
- The bundle requests the limits it needs, for example a 256 MiB binding for a 154 MB embedding,
  and refuses on an adapter that cannot grant them, as § 4.6 step 1 says.
- WebNN is an inference API whose charter excludes training, and it is not shipped; it is a later
  inference export, not a target now (§ 6).

### 2.2 Machine learning in the browser today

| Stack | What it does in a browser | Training in a browser | Source |
| --- | --- | --- | --- |
| WebLLM / MLC-LLM | LLM inference; WGSL kernels compiled ahead of time by MLC, non-kernel code in WebAssembly, the engine in a web worker; "up to 80% native performance on the same device" | none | [B1, B2] |
| WebLLM model sizes | `vram_required_MB` from 879 (Llama-3.2-1B q4f16) to 6 422 (gemma-2-9b q4f16, needs `shader-f16`); largest configuration 31 153 | — | [B3] |
| ONNX Runtime Web | Inference on WebAssembly, WebGL, WebGPU (opt-in, "only a subset" of operators) and WebNN | was CPU-only through WebAssembly, single-threaded; ORT training packages deprecated in 1.20.0; the training build is in 1.19.2's distribution and absent from the latest | [B6, B7, B8, B9] |
| transformers.js | Inference on ONNX Runtime Web; v4 has "a new WebGPU Runtime, completely rewritten in C++" | none | [B10] |
| TensorFlow.js | Inference and training on WebGL; WebGPU backend focused on inference; in maintenance | WebGL only | [B11] |
| Burn (Rust) | WebGPU through `wgpu` and CPU, compiled to WebAssembly; web examples are inference | possible, no example: "you have to write your own training loop" | [B12, B13] |
| candle (Rust) | WebAssembly examples on the CPU; no WebGPU backend on `main`; an experimental `wgpu` backend is an open pull request | none | [B14, B15] |
| Piston | Trains a 135.4 M-parameter model in a tab, about 90 tokens/s on an M1 Pro; "buffer allocation is quite slow in WebGPU"; JavaScript's garbage collector "has no concept of the full iceberg of associated VRAM" | demonstration | [B16] |
| TabGPT | Trains a 24 M-parameter GPT with forward, backward and AdamW in WebGPU shaders; matmul "roughly 1 TFLOPS f32 on Apple silicon" | demonstration | [B17] |
| Doppler | "Experimental SFT/LoRA training" for one quantised model, batch size 1 | experimental | [B21] |

WGSL matrix-multiply results: about 1 TFLOP/s on an M2 Pro, about 17 % of its peak, in a browser
[B18]; 376 GFLOP/s at 4096² in `webgpu-blas` [B20]; 1–2 % of an RTX 5090's `f32` peak for
unoptimised WGSL through native `wgpu` [B4]. The subgroup-matrix path to tensor cores is "not yet
available in stable browser releases" [B5].

### 2.3 Native GPUs

| Topic | Finding | Source |
| --- | --- | --- |
| Vulkan from Rust | `ash` 0.38.0, released 2024-04-01, binds Vulkan headers 1.3.281; `rspirv` 0.13.0 builds and parses SPIR-V for SDK 1.4.341, released 2026-03-13 | [N2, N1] |
| Software Vulkan for CI | Mesa's lavapipe "is now officially compliant with the Vulkan 1.3 specification" (2022) | [N4] |
| Vulkan on macOS | KosmicKrisp passed the Vulkan 1.4 conformance tests on Apple silicon (2026-10-01); it needs Metal 4 and macOS 26+. MoltenVK stays the solution for Intel Macs and older macOS | [N5] |
| Matrix units through Vulkan | `VK_KHR_cooperative_matrix`, ratified, revision 2; NVIDIA's `VK_NV_cooperative_matrix2`, revision 1, built on it | [N10] |
| Vulkan against CUDA, measured | llama.cpp on an RTX 3090, Llama 2 7B `Q4_0`, without flash attention: prompt processing 4 666 tokens/s on Vulkan with `coopmat2` against 5 175 on CUDA, generation 164 against 158; with flash attention 5 201 against 5 560 and 172 against 162. The two backends were measured in two community threads at different commits | [N11] |
| CUDA without the toolkit | `cudarc` 0.19.10 loads the driver at run time by default (`dynamic-loading`), but its build needs a CUDA version feature; PTX is loaded with `cuModuleLoadData`; NVRTC "is part of the CUDA Toolkit"; "Applications that compile device code to PTX will not work on older drivers" | [N6, N7] |
| AMD without Vulkan | HIP builds a code object per GPU architecture by default; ROCm's clang also offers generic AMDGCN SPIR-V (`--offload-arch=amdgcnspirv`), finalised for the concrete GPU at run time, and generic targets such as `gfx10-3-generic` | [N8] |
| Rust GPU kernel frameworks | CubeCL 0.11.2: one kernel language for CUDA, HIP, Metal, Vulkan, WebGPU and the CPU; "tensor cores are not available with the WGSL compiler (plain WebGPU) or on CPU" | [N9] |
| Native WebGPU | `wgpu` 30.0.1 (2026-08-22), over Vulkan, Metal, DirectX 12 and OpenGL | [N3, W21] |

What follows for ModelSpec [I]:

- SPIR-V on Vulkan reaches NVIDIA, AMD and Intel on Linux and Windows, Apple silicon on macOS 26
  through KosmicKrisp, and CI through lavapipe. That is M1's choice, and nothing read argues against it.
- Matrix units are reachable from SPIR-V through cooperative matrix (DESIGN.md's M4). On the
  reference GPU model, llama.cpp's Vulkan path with `coopmat2` reaches 90 % of CUDA's prefill rate
  (94 % with flash attention) and is ahead in decode [N11], though the two were measured at
  different commits. So a PTX target buys little on NVIDIA and stays last (§ 6 row 16).
- AMD's portable path is SPIR-V either way: Vulkan's, or HIP's generic AMDGCN SPIR-V. A HIP target
  would add a second runtime for the same hardware, so AMD stays on Vulkan.
- CubeCL, Burn and candle would bring their own IR and numerics in place of ModelSpec's oracle-checked
  pipeline; they are references, not back ends.

### 2.4 CPU code generation and direct machine code

| Approach | What it is | Evidence | Source |
| --- | --- | --- | --- |
| GotoBLAS and BLIS | Five loops around a register-blocked microkernel; `A` and `B` packed into contiguous panels | "near-peak performance" on many architectures; BLIS "competitive with" OpenBLAS, ATLAS and MKL; BLIS's Haswell `f32` kernel is `6 × 16` with `mc = 168`, `kc = 256`, `nc = 4080` | [C9, C3] |
| Analytical blocking | `mr`, `nr`, `kc`, `mc` from vector width, latency, port count and cache geometry, no search | matches the empirically tuned `kc` and `mc` on the machines studied | [C4] |
| JIT microkernels | libxsmm: "Just-In-Time (JIT) code specialization" for small GEMM, `(M N K)^(1/3) <= 64`; oneDNN: "JIT code generation to implement most of its functionality", with the Xbyak assembler and a batch-reduce GEMM kernel | libxsmm "highly competitive for very small problems" but "quickly gives up once the 'large' dimension exceeds about 180-240" | [C13, C14] |
| Rust encoders | `iced-x86`: encoder, decoder and formatter, "Supports all Intel and AMD instructions", AVX-512 included; `dynasm-rs`: assembly templates in macros, x86 "except for AVX-512", AArch64 "except for SVE" | | [N22, N21] |
| Compiler back ends from Rust | Cranelift: x86-64, aarch64, s390x, riscv64; LLVM 12 to 23 through `inkwell` 0.11.0, or `llvm-sys` | Cranelift about 14 % slower than an LLVM-based Wasm engine in one third-party study; no primary source on auto-vectorisation found | [N18, N19] |
| Intrinsics from generated Rust | `core::arch`, AVX-512 intrinsics stable since Rust 1.89; `wasm32` SIMD since 1.54, relaxed SIMD since 1.82; `std::simd` nightly only | | [N20, N24] |
| Pure-Rust GEMM | `matrixmultiply` 0.3.11, BLIS-style; `gemm` 0.19.0 | baselines B-mm and B-gemm in § 8.6 | [N30] |
| Schedule languages and auto-schedulers | Halide, Exo, TVM's Ansor and MetaSchedule, Triton (NVIDIA and AMD GPUs; CPUs "Under development") | Halide up to "5x faster than hand-tuned C, intrinsics, and CUDA"; its 2019 autoscheduler the "first automatic scheduling algorithm to significantly outperform human experts on average"; Exo, MKL and OpenBLAS `sgemm` all at 80–95 % of one Intel core's peak; Ansor up to 3.8× on an Intel CPU, with "more advantages on uncommon shapes and small batch sizes"; MetaSchedule 48 % end to end | [N16, N17, N14, N15, C19, C20] |
| Superoptimisation and search | STOKE (stochastic search over x86-64); equality saturation (egg, Tensat, TASO); AlphaDev; AlphaEvolve | STOKE matches or beats `gcc` at full optimisation and "in some cases, expert handwritten assembly"; Tensat up to 16 % over the state of the art; AlphaEvolve sped one kernel by 23 % | [C22, C21, C23] |
| Kernels written by language models | KernelBench, KernelBench-Verified, KernelGenBench | models match the PyTorch baseline "in less than 20% of the cases"; under stricter checking the best model reaches a "0.88x geometric mean speedup" and models "frequently engage in reward hacking" | [C24] |

The instruction sets beyond the reference machine [V for the sources, I for the consequences]:

| ISA | What it offers for `f32` | Availability | Source |
| --- | --- | --- | --- |
| AVX-512 | 32 `zmm` registers, masks, embedded broadcast | disabled or fused off on Alder Lake clients; Zen 4 splits 512-bit operations over 256-bit units; Zen 5 desktop and EPYC 9005 have the full datapath, Ryzen AI 300/400 do not | [C16] |
| AMX | 8 tiles of 16 × 64 B; `bf16` and `int8` inputs, `f32` accumulation with DAZ and FTZ forced | no `f32`-input path: AMX-TF32 was removed before any hardware shipped | [C15] |
| Neon | 32 × 128-bit registers; Apple M1's Firestorm issues 4 `FMLA` per cycle, 32 `f32` FLOP per cycle | every AArch64 core | [C17, C18] |
| SVE, SME, SME2 | vector length 128 to 2048 bits; SME's ZA tile array and outer-product `FMOPA` | Apple M4: one P-core's `FMOPA` measured at 2 009 `f32` GFLOP/s, against 113 for Neon on the same core | [C17, C18] |

What follows for ModelSpec [I]:

- The established way to reach near peak for dense `f32` matrix multiply on a CPU is a template
  microkernel, analytical blocking and a JIT encoder (BLIS, libxsmm, oneDNN). § 8 takes that route.
  ModelSpec adds an exact mode, fusion with the consumer, and static shapes known at compile time.
- Search pays on uncommon shapes and fusion [C20], which is where M0's small layers sit; a search
  result is goldenable only when pinned, so § 8.4 uses search to check the analytical model, not to
  choose.
- Language-model kernels need an independent check to be trusted [C24]. ModelSpec's oracle is that
  check, bit for bit in exact mode; an agent proposing schedules that the oracle verifies is a
  later experiment, not part of § 8.
- AMX does not help an `f32` training step; AVX-512 is not on the reference machine; SME is the
  largest `f32` step on any CPU read about here, which makes AArch64 the most interesting second
  ISA once a machine exists (row 10 of § 6).

### 2.5 Graph and weight exports

| Format | What it carries | Status | Source |
| --- | --- | --- | --- |
| ONNX | An operator graph with a versioned IR and operator sets; `TrainingInfoProto` can describe a training step | ONNX 1.23 is IR version 14, opset 28; ONNX Runtime's training packages are deprecated since 1.20.0, and 1.19.2 was the last release to publish them | [N25, N26, B7] |
| ONNX from Rust | `ort` binds ONNX Runtime (2.0.0-rc.15, no stable 2.0); `tract` is "Sonos' neural-network inference engine" in pure Rust | both consume ONNX; no maintained Rust crate that writes ONNX was found | [N27; I: search] |
| StableHLO | XLA's portable operator set, serialised through the versioned VHLO dialect, each version tied to an MLIR bytecode format version | 5 years backward and 2 years forward compatibility | [N13] |
| IREE | Compiles StableHLO and other MLIR inputs | targets include `llvm-cpu`, `vulkan-spirv`, `cuda` and `metal-spirv`; AMD GPUs and `webgpu-spirv` experimental | [N12] |
| GGUF | llama.cpp's weight file: magic, version, tensor count, metadata key-value pairs, tensor data; little-endian | version 3 | [N28] |
| safetensors | An 8-byte little-endian header length, a JSON header, then raw tensor bytes | Rust crate 0.8.0 by its authors | [N29] |

What follows for ModelSpec [I]:

- safetensors is a few hundred lines from DESIGN.md § 6.2's checkpoint, which already has a JSON
  manifest and a little-endian payload; it is the cheapest useful export (§ 6 row 4).
- ONNX is the widest inference export, ONNX Runtime Web included [B9]. Training graphs in ONNX have
  no maintained runtime left, so the export is forward only. Without a Rust writer the exporter
  generates its protobuf types from `onnx.proto`.
- StableHLO is the one format that can carry a whole training step to another compiler (XLA,
  IREE). Without a Rust writer it is large work, and it waits for a user who needs it.

## 3. Does training on a GPU in a browser make sense?

**Verdict.** Yes, for a narrow job: small models, zero-install demonstration and teaching, private
fine-tuning of small parameter sets on the user's device, and a second GPU stack that checks a
`.mspec` step trains anywhere. No, for throughput, bit-exact results, long unattended runs, or
models whose state is over about 1.5 GiB. For ModelSpec it costs little beyond browser
inference, because the compiler already emits the whole step: backward and SGD are more kernels of
the same WGSL emitter and the same executor. [I, from the numbers below]

### 3.1 Memory

Training state per parameter in `f32`: the value and its gradient for SGD (8 B), plus two moment
estimates for AdamW once M3 adds it (16 B). Activations come on top and are not counted here.

| Model | Parameters | SGD state | AdamW state | Largest single tensor | Against the 128 MiB default binding [W5] |
| --- | --- | --- | --- | --- | --- |
| `examples/tiny_mlp.mspec` | 2 112 [V] | 16.9 KB | 33.8 KB | 2 048 elements, 8 KiB | fits |
| TabGPT's browser GPT (10 layers, width 384) [B17] | 24 M | 192 MB | 384 MB | | fits per tensor; total over the 256 MiB default `maxBufferSize` only if packed into one buffer |
| GPT-2 small | 124 M | 0.99 GB | 1.98 GB | token embedding 50 257 × 768 = 154.4 MB | exceeds 134.2 MB: needs a raised limit or a split tensor |
| 1 B parameters | 1 000 M | 8 GB | 16 GB | | beyond any documented tab budget |

[I, arithmetic; GPT-2 small's `vocab_size` 50 257, `n_embd` 768 and `n_layer` 12 from its
published configuration [R5]]

Adapters raise the defaults when asked: `maxStorageBufferBindingSize` is at least 1.5 GiB on 79 %
of web3dsurvey's reports and at least 2 GiB on 16 %; `maxBufferSize` is at least 2 GiB on 79 %
[W8]. No browser publishes a per-tab GPU memory cap [W5, W14: none stated]; one study reports an
iOS Safari tab "limited to <500 MB" [B5], and a 135.4 M-parameter model trained in a tab on a
16 GB M1 Pro [B16]. The practical ceiling for training state is therefore about 1.5 GiB on most
desktop adapters and much less on phones. [I]

### 3.2 Compute

Training costs about `6N` floating-point operations per token, `N` the non-embedding parameter
count ("C≈6N floating point operators per training token" [R4]).

| Case | Rate | Time for 10⁹ tokens |
| --- | --- | --- |
| GPT-2 small, `N` ≈ 85 M (`12 · n_layer · d_model²` [R4] with 12 layers of width 768 [R5]), at 1 TFLOP/s, the best browser `f32` matmul reported [B18, B17] | 0.51 GFLOP per token, about 1 960 tokens/s | 5.9 days |
| Piston's 135.4 M-parameter model in a tab on an M1 Pro, measured end to end [B16] | 90 tokens/s, about 73 GFLOP/s, about 1.4 % of that GPU's `f32` peak [B16, B22; derived] | 129 days |

[I, arithmetic from the cited rates]

Small models are launch-bound, not compute-bound. A WebGPU dispatch costs 32.8 µs in Chrome on
Linux, 58.7 µs in Chrome on Windows, 31.7 µs in Safari and about 1 040 µs in Firefox [B4]. One
`tiny_mlp` step is about 0.81 MFLOP (`2·64·32·64 + 2·64·64` forward, three times that for the
step [R4, I]) in roughly ten dispatches, so dispatch overhead alone caps it near 3 000 steps per
second, about 2.5 GFLOP/s, in Chrome on Linux. The same arithmetic is about 63 000 cycles, 12 µs
at 5.3 GHz, for one core at the 12.8 FLOP per cycle the exact kernel of § 8 aims for, which it
will not reach at these shapes (§ 8.3). At M0 sizes the CPU wins. [H: the operator's first browser
measurement checks it]

### 3.3 Numerics

- WGSL lets an implementation reassociate, fuse, flush denormals in its arithmetic operations and
  assume finite values [W6a–W6e]. A browser result is therefore judged against computed bounds per
  step and in lockstep (§ 5.3, § 5.4), never bit for bit.
- WGSL atomics are `i32` and `u32` only [W6f]; `f32` atomics are an open issue [W7]. This costs
  ModelSpec nothing in `deterministic` mode, which refuses atomic accumulation (`Unordered`)
  anyway; fixed two-stage reductions are what the WGSL schedule uses in either case. [I]
- `f32` is the only universal float type. `f16` needs the `shader-f16` feature [W9], enabled on
  Linux NVIDIA only from Chrome 155–156 [W11]; `bf16` is not available [B5]. M0 is `f32`-only, so
  nothing is lost now; M3's mixed precision would be limited in the browser.
- Timestamp queries are quantised to 100 µs in Chrome [W12], so kernel-level profiling in a tab is
  coarse; per-step wall time is fine.

### 3.4 The tab is not a server

- The user agent loses a device whenever it "needs to revoke access to a device" [W5]; Memory
  Saver "deactivates tabs that you aren't currently using", and such a tab reloads when opened
  [W14]; hidden-tab timers run once per second, once per minute after five minutes hidden when
  further conditions hold, and animation frames wait until the page is visible [W13].
- So the executor checkpoints to the Origin Private File System [W25] every `n` steps and resumes
  from the last one after a reload or a lost device (§ 4.6). It runs the step loop in a dedicated
  worker, where WebGPU is available [W24]; whether that escapes background throttling is not
  documented in what was read [H].

### 3.5 What exists

No major stack ships GPU training in the browser: ONNX Runtime Web trained on the CPU through
WebAssembly only and its training packages were deprecated in 1.20.0 [B6, B7]; TensorFlow.js
trains on WebGL and says of its WebGPU backend "At this point we are focused on making inference as
fast as possible" [B11]; transformers.js and WebLLM are inference-only [B10, B2]. Demonstrations
exist: Piston, whose author calls the browser "a punishingly impractical place" for training
[B16]; TabGPT [B17]; experimental LoRA in Doppler [B21]. Burn's maintainer: "we already support
training on the web, but you have to write your own training loop" [B13]. A ModelSpec browser
target would be one of very few that train, and the only one whose step is verified against a CPU
oracle. [I]

### 3.6 Browser inference

Inference is the mature case. WebLLM reports that it "can retain up to 80% native performance on
the same device" (Apple M3 Max: Llama-3.1-8B decode 41.1 against 57.7 tokens/s natively, 71 %)
[B1]. The gap is larger elsewhere: on NVIDIA, native CUDA and Vulkan "outperform WebGPU by up to
10×" in prefill and 2.5× in decode [B5], WGSL reaches 1–2 % of an RTX 5090's `f32` peak in an
unoptimised study and "Tensor cores are not accessible via WGSL" [B4], and in that study WebLLM's
shaders failed to compile in Chrome on Linux with Vulkan [B4]. For ModelSpec, browser inference is
the forward subgraph of the same bundle (§ 4.6) and comes first in the order (§ 6).

## 4. Target architecture

### 4.1 Two branch points

Targets fall into two families, and they leave the pipeline at different places.

- **Graph targets** keep the operations and drop the loops: ONNX, StableHLO, a WebNN graph, and
  weight formats (safetensors, GGUF). They branch at the Tensor IR (inference) or the optimised
  Training IR (a whole step), before any schedule exists.
- **Code targets** produce loops for a machine: SPIR-V, WGSL, x86-64 and AArch64 machine code,
  generated Rust, WebAssembly. They share everything up to a target-neutral Kernel IR and branch at
  scheduling, because a schedule (tiles, lanes, workgroups, memory spaces) is where targets
  genuinely differ.

```text
.mspec -> parse, check -> Tensor IR -> Training IR -> optimised Training IR      DESIGN.md § 3
                              |                          |
                              +-- B1 graph export -------+--> ONNX (forward), StableHLO (step),
                              |                               WebNN graph, safetensors / GGUF
                              v
                   Kernel IR: loops, index maps, reductions with an explicit policy
                              |   reference interpreter (f32, f64)
                   B2 schedule for a target family
                              v
                   Scheduled IR: tiles, lanes, threads or workgroups, memory spaces
                              |   reference interpreter (f32, f64)
             +----------------+-----------------+--------------------+
             v                v                 v                    v
        SPIR-V (rspirv)   WGSL (text)     x86-64 / AArch64      Rust source, Wasm
             |                |           machine code
        Vulkan (ash)     WebGPU (browser)  JIT or object file
        M1               wgpu natively for CI
                              |
                   Execution DAG (plan) + modules -> CompiledArtifact / web bundle
```

The diagram makes DESIGN.md's single "target lowering" step (§ 3) into two named levels. DESIGN.md
already defines Kernel IR as "low-level loops, index expressions, dispatch dimensions,
workgroup/subgroup mappings, buffer spaces, layouts, barriers, and lifetimes" (§ 3.1). This design
splits that definition: the loops, index expressions and reductions are the target-neutral Kernel
IR; dispatch dimensions, workgroup mappings and buffer spaces are the Scheduled IR, which is what
DESIGN.md's illustrative `KernelBackend::compile(&ScheduledProgram)` (§ 3.2) already takes.

### 4.2 Kernel IR and Scheduled IR

**Kernel IR** is a list of kernels. A kernel is a perfect or imperfect loop nest over static index
spaces with affine loads and stores on logical buffers, scalar operations of the element type, and
reductions. A reduction carries its policy (§ 5.2) as data, so order is part of the program and
not an accident of a loop. Illustrative print for the first layer of `examples/tiny_mlp.mspec`
fused with its ReLU (`[64, 32] x [32, 64]`):

```text
kernel %k0 (x: f32[64, 32], w: f32[32, 64]) -> h: f32[64, 64]
  for i in 0..64, j in 0..64:
    %acc = reduce sequential k in 0..32: mul(x[i, k], w[k, j])
    h[i, j] = select(gt(%acc, 0.0), %acc, 0.0)
```

`select(gt(acc, 0), acc, 0)` is the interpreter's `relu_where` written out; a lowering that wrote
`max(acc, 0)` would change `relu(-0.0)` or `relu(NaN)` on some targets (§ 8.3, § 2.1).

**Scheduled IR** is Kernel IR plus a schedule per kernel: loop order, tile sizes per level, the
vectorised axis and its width, unrolling, the parallel axes (threads on a CPU, workgroups and
invocations on a GPU), the memory space of each buffer (global, workgroup-shared, registers), and
packing copies. Two classes of transformation are kept apart:

| Order-preserving (allowed in `deterministic` mode under policy `Sequential`) | Order-changing (changes the reduction policy) |
| --- | --- |
| Tiling and reordering non-reduction axes | Split-K, or any partition of a reduction axis whose partials are combined afterwards |
| Blocking the reduction axis while carrying the partial sum through memory in ascending order | Vectorising along the reduction axis |
| Vectorising or parallelising non-reduction axes | Tree, pairwise or workgroup reductions |
| Packing and layout copies | Atomic accumulation |
| Fusing an elementwise epilogue | Fused multiply-add (changes rounding, not order) |

The scheduler may apply an order-changing transformation only if the selected numeric contract
allows the policy it produces (§ 5.1), and it records that policy in the Scheduled IR so the
oracle can reproduce it.

Both levels get a reference interpreter in the CPU runtime, generic over `f32` and `f64` like the
Tensor IR interpreter. That gives translation validation per stage: the Tensor IR interpreter, the
Kernel IR interpreter and the Scheduled IR interpreter must agree bit for bit whenever the
reduction policy is the same, and a target must agree with the Scheduled IR interpreter running the
target's policy. A disagreement then names the stage that introduced it. This is DESIGN.md's
invariant 7 ("Testable at every lowering stage") applied below the Tensor IR. [I]

### 4.3 Crates and dependency rules

| Crate | Holds | Depends on (new) | Must not depend on |
| --- | --- | --- | --- |
| `modelspec-core` | Language, Tensor IR, Training IR, Kernel IR, Scheduled IR, verifiers, the `Target` trait and `NumericContract` | nothing new | any GPU crate, any code generator |
| `modelspec-codegen` (new) | Schedulers and emitters: SPIR-V (`rspirv`), WGSL (text), x86-64 (`iced-x86`), plan serialisation, graph exporters | `rspirv`, `iced-x86`, `serde` | `ash`, `vulkano`, `wgpu`, `wgpu-hal`, `gpu-allocator`, `vk-mem`, `memmap2`, any FFI to a device library |
| `modelspec-runtime` | CPU interpreters (all three levels), executors behind features: `vulkan` (`ash`), `jit-x86` (`memmap2`), `wgpu-check` (conformance only) | per feature | nothing in the default feature set beyond today's |
| `modelspec-cli` | `--target <id>`, `--emit <listing>` | | |
| `web/` (new, TypeScript) | Vue 3 + TypeScript + Vite application: WebGPU executor, bundle loader, training UI | pnpm packages | anything that runs outside the browser |

Emission moves out of the runtime. DESIGN.md § 7 places `spirv/{emit,validate}.rs` in
`modelspec-runtime`; a compile on a machine without Vulkan (CI, a build server, the compiler
compiled to `wasm32` in a browser tab) must still be able to emit SPIR-V or WGSL, and the
`no_vulkan` guard extends naturally to a `codegen-no-device` guard on the new crate. [I]

### 4.4 Rust interfaces (illustrative)

The `Target` side is pure: it turns a program into bytes and text and never touches a device. The
executor side owns a device. Keeping them apart lets the same emitter serve a native executor, a
browser executor written in TypeScript, and an ahead-of-time bundle built on a machine with no GPU.

```rust
/// A stable target name: `cpu-ref`, `x86_64-v3-exact`, `x86_64-v3-fma`, `spirv-vk13`,
/// `wgsl-webgpu`, `onnx-opset28`, ... Recorded in the artifact and its hash.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TargetId(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompileFlags {
    pub deterministic: bool,
    pub fast_math: bool,
}

pub trait Target {
    fn id(&self) -> TargetId;

    /// What the device or its recorded profile offers (DESIGN.md § 0 invariant 6): vector width,
    /// FMA, f16, subgroups, shared-memory size, buffer limits, float controls, atomics.
    fn capabilities(&self) -> &Capabilities;

    /// The floating-point behaviour of code this target emits under `flags` (§ 5.1). Recorded in
    /// the artifact; it decides how conformance compares the target with the oracle.
    fn numeric_contract(&self, flags: CompileFlags) -> NumericContract;

    /// Choose a schedule; refuse what the target cannot run with an E-TARGET diagnostic.
    fn schedule(&self, kernels: &KernelProgram, flags: CompileFlags)
        -> Result<ScheduledProgram, TargetError>;

    /// Pure emission: modules, the execution plan, and listings for golden files.
    fn emit(&self, program: &ScheduledProgram) -> Result<TargetOutput, TargetError>;
}

pub struct TargetOutput {
    pub modules: Vec<EmittedModule>,      // SPIR-V words, WGSL text, machine code + relocations
    pub plan: Vec<PlanStep>,              // the Execution DAG (spec/domains/artifact.yaml)
    pub listings: Vec<Listing>,           // `.kernel.ir`, `.sched.ir`, `.spvasm`, `.wgsl`, `.s`
}

/// Graph targets skip Kernel IR and implement this instead.
pub trait Exporter {
    fn id(&self) -> TargetId;
    fn export(&self, step: &TrainingStep, scope: ExportScope) -> Result<Vec<u8>, ExportError>;
}
```

The executor is DESIGN.md § 6's `DeviceRuntime`, unchanged in shape (`load`, `execute_step`), with
one implementation per execution environment: the CPU interpreter, Vulkan (`ash`), the x86-64 JIT,
and, outside Rust, the browser executor. The browser executor implements the same observable
contract in TypeScript and is held to it by the same conformance records (§ 5.4).

`NumericContract` is the piece every later section relies on:

```rust
pub struct NumericContract {
    pub reduction: ReductionPolicy,
    pub contraction: Contraction,     // Separate | Fused
    pub denormals: Denormals,         // Preserve | FlushToZero | Unspecified
    pub specials: Specials,           // signed zero, infinities and NaN preserved or not
    pub elementwise: Elementwise,     // CorrectlyRounded, or an ulp bound per operation
    pub repeatability: Repeatability, // BitExact run to run on one device and driver, or Unspecified
}

pub enum ReductionPolicy {
    /// DESIGN.md § 2.2: ascending, sequential, starting from the first term.
    Sequential,
    /// `lanes` sequential partial sums, term `t` to lane `t mod lanes`, combined by `combine`.
    Strided { lanes: u32, combine: Combine },
    /// Fixed-size workgroup partials, then a final pass (DESIGN.md § 5.2 "2-stage reduction").
    TwoStage { group: u32, within: Box<ReductionPolicy>, across: Box<ReductionPolicy> },
    /// Order not fixed (atomics, dynamic split-K). Refused when `deterministic` is set.
    Unordered,
}
```

Every policy except `Unordered` is a deterministic function of the input, so the oracle can run it
(§ 5.2). `Unordered` exists so a fast-math GPU target can say what it does.

### 4.5 Artifact changes

- `Backend { Cpu, Vulkan }` becomes a `TargetId`. The existing two values map to `cpu-ref` and
  `spirv-vk13`.
- `KernelModule.spirv: Bytes` becomes `code: Bytes` with `format: ModuleFormat` (`SpirV`, `Wgsl`,
  `X86_64`, `Aarch64`, `Wasm`). A WGSL module's bytes are its UTF-8 text.
- The artifact records the `NumericContract` it was compiled under, and the contract is part of the
  artifact hash, as `fast_math` and `deterministic` are today (DESIGN.md § 3.3).
- `device_profile_id` stays Vulkan-only. A CPU machine-code artifact records a `CpuProfile` (ISA
  level, features, cache sizes), and a browser run records the WebGPU adapter it saw in its
  conformance record, not in the artifact, because the bundle is built before any adapter is known.
- `PlanStep` is unchanged: a CPU kernel's parallel grid fits `workgroups: Extent3D`. The WebGPU
  executor maps a `Barrier` to the boundary between two dispatches, each its own usage scope
  [W6g]. Neither specification was found to state that this alone makes one dispatch's storage
  writes visible to the next (§ 2.1), so the executor's conformance suite starts with a
  write-then-read pair of dispatches on every adapter it records, and a `Barrier` becomes a
  compute-pass boundary if that pair ever fails. [H]

### 4.6 The web bundle and the browser executor

The browser path is ahead-of-time: the Rust compiler builds a **web bundle**, and a TypeScript
executor runs it. The bundle is a directory, or one archive of it:

| File | Content | Typed by |
| --- | --- | --- |
| `manifest.json` | The `CompiledArtifact` manifest with target `wgsl-webgpu`: parameters, data bindings, numeric contract, required WebGPU features and limits | ESS `modelspec.artifact` |
| `kernels/<id>.wgsl` | One WGSL module per kernel | — |
| `plan.json` | The Execution DAG | ESS `PlanStep` |
| `state.msckpt` | Initial parameters, or a checkpoint to resume or to run inference from | DESIGN.md § 6.2 |

The executor, in `web/` as a Vue 3 application:

1. Requests an adapter and device with exactly the features and limits the manifest names, and
   refuses with the E-TARGET code the manifest gives when the adapter lacks one.
2. Creates one storage buffer per plan buffer and one compute pipeline per module.
3. Per step, encodes every Dispatch of the plan into one command encoder and submits it.
4. Reads the loss back with `mapAsync` every `n` steps, not every step, so the GPU queue is never
   drained for telemetry.
5. Writes checkpoints in the `.msckpt` format to the Origin Private File System, so a reloaded tab
   resumes at the last checkpoint instead of at step 0.

The TypeScript types for the manifest, the plan and the checkpoint header are generated from the
Rust definitions (which follow the ESS types) by a Rust generator in the repository, and a drift
check in `task ci` fails when the generated files differ. The `.msckpt` reader stays a few dozen
lines: the format is a fixed header, a JSON manifest and a little-endian payload (DESIGN.md § 6.2).

Later, the compiler itself can run in the tab. The core depends only on `logos` and `thiserror`
[V, `crates/modelspec-core/Cargo.toml`], and the planned codegen dependencies are pure Rust, so a
`wasm32-unknown-unknown` build is plausible [I]; it turns the bundle step into an in-browser
playground. It is Rust running in a browser, so it needs the exception that decision D2 option B
names. The Vue application stays the
user interface either way; the compiler would be a wasm module it calls.

## 5. Numerics, determinism and conformance

### 5.1 The numeric contract of each target

What each environment permits decides what a target can promise. The rows below are the
contracts this design assigns in `deterministic` mode; `fast_math` relaxes them as noted.

| Target | Reduction | Contraction | Denormals | Specials | Elementwise | Repeatability | Bit-exact with the oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cpu-ref` | `Sequential` | `Separate` | `Preserve` | preserved | correctly rounded [R1] | `BitExact` | is the oracle |
| `x86_64-v3-exact` | `Sequential` (or `Strided { 8 }`, recorded) | `Separate` | `Preserve` (MXCSR default) | preserved | correctly rounded | `BitExact` | yes [H, § 8.7 S1] |
| `x86_64-v3-fma` | `Sequential` per element | `Fused` | `Preserve` | preserved | correctly rounded | `BitExact` | with the oracle running `Fused` [I] |
| `spirv-vk13` | `Sequential` or `TwoStage`, fixed | `Separate` via `NoContraction` | `Unspecified` on the reference GPU | preserved where `SignedZeroInfNanPreserve` is supported | `+ − ×` correctly rounded, `÷` 2.5 ULP [K1] | `BitExact` [H] | only on inputs and intermediates without subnormals [I] |
| `wgsl-webgpu` | as emitted, but WGSL permits reassociation | WGSL permits fusion | may flush | not guaranteed: finite math may be assumed | `+ − ×` correctly rounded, `÷` 2.5 ULP for `|y|` in `[2⁻¹²⁶, 2¹²⁶]` [W6d] | `Unspecified` | no; bound and lockstep only |

The two GPU rows rest on what the specifications allow, quoted in § 2.1 and [K1]:

- **Vulkan.** By default "Implementations may rearrange floating-point operations using any of the
  mathematical properties", and the flags `AllowContract` and `AllowReassoc` are assumed; "This
  rearrangement can be prevented for particular operations by using the NoContraction
  decoration" [K1]. So the SPIR-V emitter decorates every floating-point operation `NoContraction`
  in `deterministic` mode. Denormals: "By default any denormalized floating-point value input into
  a shader or potentially generated" may be flushed to zero; `DenormPreserve` would forbid that,
  but "If shaderDenormPreserveFloat32 is VK_FALSE, then DenormPreserve for 32-bit floating-point
  type must not be used" [K1], and on the RTX 3090 it is false [V, § 1.4]. Specials: without
  `SignedZeroInfNanPreserve` "the entry point assumes NSZ, NotInf, and NotNaN"; the reference GPU
  supports it [V], so the emitter declares it.
- **WebGPU.** WGSL states that "An implementation may reassociate operations", may fuse operations
  when the result is "at least as accurate", may flush subnormal inputs and outputs of the
  operations in its accuracy table, and "may assume that overflow, infinities, and NaNs are not
  present" [W6a–W6e]. No WGSL attribute turns any of this off [I: none was
  found in the specification]. The WGSL target therefore promises nothing bit for bit, and its
  conformance is the bound of § 5.3 per step plus lockstep (§ 5.4). A non-finite value anywhere
  in the oracle's run is reported as outside the WGSL contract, not compared.

What the comparator does with `Unspecified` denormals is in § 5.4. The practical consequence for
test data: conformance inputs are drawn so that no input and no exact intermediate of the
reference shapes is subnormal [I]; a separate denormal suite checks only that the result stays
within `2⁻¹²⁶` of the oracle.

### 5.2 A parameterised oracle

Today the interpreter has one reduction order. The proposal is to give `sum` and `matmul` a
`ReductionPolicy` argument and a `Contraction` argument, defaulting to `Sequential` and
`Separate`, so today's behaviour and every golden value stay as they are.

- `Strided`, `TwoStage` and any fixed tree are a few lines each over the same element type.
- `Fused` uses `f32::mul_add`, which "is guaranteed to be the rounded infinite-precision result.
  It is specified by IEEE 754 as `fusedMultiplyAdd` and guaranteed not to change" [R2].
- `FlushToZero` is a post-operation map that replaces subnormal results with zero of the same
  sign; `Unspecified` denormals are handled by the comparator (§ 5.4), not by the oracle.

With that, every deterministic target has a bit-exact reference: the oracle running the target's
own contract. A tolerance is needed only between two different contracts, and that tolerance is
computed, not chosen (§ 5.3).

### 5.3 Tolerances from a rounding-error bound

For an inner product of length `n` in floating point with unit roundoff `u`,
`|fl(xᵀy) − xᵀy| ≤ γ_n |x|ᵀ|y|` with `γ_n = (1+u)^n − 1 ≤ nu/(1 − nu)` [R3]. For `f32`,
`u = 2⁻²⁴`. The same constant bounds recursive summation; for blocked or tree orders the classical
analysis gives a smaller depth-dependent constant, so `γ_n` bounds all of them [I: standard result,
not re-read today].

The comparator therefore does not compare two `f32` results with each other. It compares each
target result with the `f64` oracle, whose own error is about `2⁻²⁹` of the `f32` bound and
negligible, and accepts an element when `|ŷ − y_f64| ≤ κ · b`, where `b` is the element's bound
and `κ` a fixed safety factor for the neglected second-order terms. `κ = 2` is the proposed
starting value; the existing `f32` gradient data (`TOL_F32 = 1e-3` against `f64`) is the first
evidence to check it against. [I]

The bound `b` is computed by a **shadow interpreter**: the `f64` interpreter carries a magnitude
and a first-order error bound beside every value, with one rule per operation in the operator
registry. DESIGN.md § 2.2 already asks the registry to define each operator's "precision
behavior"; these rules are that definition. [I]

| Operation | First-order error rule (`e` = bound, `u` = unit roundoff of the target type) |
| --- | --- |
| `a ± b` | `e_a + e_b + u·|a ± b|` |
| `a * b` | `|a|·e_b + |b|·e_a + u·|a·b|` |
| `matmul(A, B)`, contraction length `K` | `|A|·E_B + E_A·|B| + γ_K·|A|·|B|` (one rounding fewer per term when fused) |
| `relu(a)` | `e_a` (1-Lipschitz) |
| `relu_grad(a, g)` | `e_g` where `|a| > e_a`; where `|a| ≤ e_a` the element may flip between `0` and `g`, so `|g| + e_g`, and it is reported as unstable |
| `sum` over `n` terms | `Σ e_i + γ_(n−1)·Σ|x_i|` |
| `mean`, `mse` | the sum's bound divided by `n`, plus `u` times the result |
| SGD `p − lr·g` | `e_p + |lr|·e_g + u·(|lr·g| + |p − lr·g|)` |
| any result under `Denormals::Unspecified` | add `2⁻¹²⁶` (the smallest normal `f32`) |

Run-level comparisons do not chain these bounds over hundreds of steps; they would grow without
limit. The run level uses lockstep comparison instead (§ 5.4).

### 5.4 Conformance levels

| Level | Unit compared | Same contract as oracle | Different contract |
| --- | --- | --- | --- |
| Kernel | One kernel's outputs | Bit-exact | Within the shadow bound per element |
| Step | Loss, every gradient and every updated parameter of one train step | Bit-exact | Within the shadow bound; unstable `relu_grad` elements listed, not failed |
| Lockstep | Step `s+1` started from the target's own state after step `s`, for every `s` of a run | Bit-exact | Within the step bound, from the target's state |
| Run | Final loss and parameters after the configured steps | Bit-exact | DESIGN.md § 9 gate 4: "verify loss decreases and parameters approach a known solution" |
| Repeat | The same target, twice | Bit-exact if the contract says `BitExact` | Bit-exact if the contract says `BitExact` |

Lockstep is what makes long GPU runs checkable without a tolerance that grows with the step count:
each step is compared from the same starting state, so an error cannot hide behind earlier
divergence. [I]

Comparison rules common to every level:

- NaN compares as a class: any NaN equals any NaN, payload and sign ignored [R1].
- Signed zero compares exactly when both contracts preserve specials, and as zero otherwise.
- Under `Denormals::Unspecified`, a subnormal on either side compares as zero of either sign.
- Shapes include 1, primes, and every tile and workgroup size minus one, equal and plus one, as
  DESIGN.md § 9 gate 5 asks ("including edges not aligned to workgroup/tile sizes").

Each run writes a **conformance record**, an ESS type (§ 12): target id, the device or adapter
profile, artifact hash, oracle contract, level, shapes, the largest absolute error, the largest
ratio of error to bound, the unstable elements, the verdict, the UTC time and the command that
produced it. The record is how a browser run on the operator's machine, which CI cannot repeat,
still leaves evidence.

### 5.5 Golden files

Emitters are deterministic, so their output is golden text beside today's IR goldens
(`crates/modelspec-cli/tests/golden/`):

| Golden | Produced by | Notes |
| --- | --- | --- |
| `<example>.kernel.ir` | Kernel IR printer | target-neutral |
| `<example>.<target>.sched.ir` | Scheduled IR printer | under a pinned capability profile |
| `<example>.spirv-vk13.spvasm` | `rspirv` disassembly | plus the blake3 of the binary words |
| `<example>.wgsl-webgpu.wgsl` | the WGSL emitter | the text is the module |
| `<example>.x86_64-v3-exact.s` | `iced-x86` formatter, Intel syntax, symbolic labels | no absolute addresses |
| `<example>.<target>.plan.json` | plan serialiser | ESS-typed |

A schedule that depends on capabilities is golden only under a pinned profile: a fixture
`DeviceProfile` for the reference GPU, a static `x86_64-v3` CPU profile, and the WebGPU default
limits. A golden never depends on the machine running the test.

Two mechanical checks travel with the goldens: every emitted WGSL module must parse and validate
with `naga` inside `cargo test`, and every exact-mode x86-64 module is decoded with `iced-x86` and
must contain no fused multiply-add instruction. The second turns the exactness contract into a
property CI checks on every build. [I]

### 5.6 Where conformance runs

| Check | Runs where | Needs | Level |
| --- | --- | --- | --- |
| Kernel IR and Scheduled IR interpreters against the Tensor IR interpreter | CI, `cargo test` | nothing | Kernel, Step: bit-exact under the same policy |
| Emitted WGSL parses and validates with `naga`; emitted SPIR-V validates | CI | `naga` (Rust) | golden plus validation |
| Exact-mode x86-64 modules contain no FMA opcode | CI | `iced-x86` decoder | golden |
| x86-64 JIT kernels against the oracle | CI on x86-64 runners; the test reads CPUID and records `skipped`, never `passed`, without AVX2 | an x86-64 runner | Kernel, Step |
| Vulkan executor on a software device | CI with Mesa's lavapipe, which is conformant to Vulkan 1.3 [N4] | lavapipe in the CI image; it is not installed on the reference machine [V] | Kernel, Step, Run |
| WGSL modules executed natively | CI through `wgpu` on lavapipe | `wgpu` behind the `wgpu-check` feature | Kernel, Step against bounds |
| Vulkan on the RTX 3090 | the operator | the device | all levels; records kept |
| WebGPU in Chrome on the reference machine | the operator | Chrome 150 [V] and an adapter Chrome enables (§ 2.1) | Step, Lockstep, Run; records kept |
| x86-64 benchmarks | the operator, pinned core, quiet machine | `perf_event_open` [V] | benchmark records (§ 8.6) |

The native `wgpu` run is not the browser: Chrome compiles WGSL with its own Dawn and Tint stack
[W23], and `wgpu` compiles it with `naga` [W22]. It checks the WGSL emitter's arithmetic; only an
operator run in a browser checks the browser. A record from that run carries the adapter
profile, so a later failure on another machine is a different row, not a contradiction.

## 6. Target matrix and order

Effort for one engineer working with agents: **S** up to a week, **M** two to four weeks, **L** one
to three months, **XL** more [I]. Branch: B1 graph export, B2 code target (§ 4.1). Rows above the
line are planned in this order; rows below it stay research until a named trigger.

| # | Target (id) | Use case | Branch | Effort | Risk | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Kernel IR, Scheduled IR, interpreters | Every code target | B2 | M | Low | M0 |
| 2 | Vulkan, `spirv-vk13` (M1) | Native GPU training and inference | B2 | L | Medium: driver variance, denormals | 1; `ash`, `rspirv` [N1, N2] |
| 3 | x86-64 machine code, `x86_64-v3-exact` and `-fma` (§ 8) | Exact, fast CPU training; the experiment | B2 | M, two-week budget | Medium: executable memory | 1; `iced-x86` [N22] |
| 4 | safetensors weights | Exchange weights with other frameworks | B1 | S | Low | DESIGN.md § 6.2 checkpoints; `safetensors` [N29] |
| 5 | **Browser inference**, `wgsl-webgpu`, forward subgraph | Zero-install inference and demonstrations (§ 3.6) | B2 | M emitter + M TypeScript executor | Medium: browser and driver variance, WGSL numerics (§ 5.1) | 1, 2's GPU schedules; `naga` for validation [W22] |
| 6 | **Browser training**, `wgsl-webgpu`, whole step | Small models, teaching, private fine-tuning (§ 3) | B2 | S on top of 5, plus OPFS checkpoints and lifecycle | Medium to high: tab lifecycle, memory ceiling (§ 3.1, § 3.4) | 5 |
| 7 | ONNX, forward only, `onnx-opset<n>` | Inference wherever ONNX Runtime runs, ORT Web included [B9] | B1 | M: no maintained Rust ONNX writer, so protobuf from `onnx.proto` [N25, N27] | Low to medium: operator coverage | Tensor IR |
| | *planned line* | | | | | |
| 8 | Generated Rust with `core::arch` intrinsics | B-intrin control of § 8.6; portable CPU code through `rustc` | B2 | S to M | Low | 1; built inside experiment 3 |
| 9 | AVX-512, `x86_64-v4` | Wider CPU kernels (32 registers, masks) | B2 | M | No AVX-512 on the reference machine [V] | 3; trigger: hardware |
| 10 | AArch64 NEON, SVE, SME | ARM servers, Apple silicon | B2 | L | No AArch64 machine | 3; trigger: hardware |
| 11 | Metal | macOS | through 2 | S if 2 is portable | No Mac inventoried | 2 on KosmicKrisp [N5]; trigger: a Mac |
| 12 | StableHLO, whole step | XLA and IREE, whose targets include `llvm-cpu`, `vulkan-spirv`, `cuda` and `metal-spirv` [N12, N13] | B1 | L: MLIR bytecode, no Rust writer found | Medium | Training IR; trigger: a user of XLA or IREE |
| 13 | WebNN graph | Browser NPUs, inference only | B1 | M | High: not shipped [W19, W20] | 7; trigger: stable release |
| 14 | WebAssembly SIMD CPU kernels | CPU fallback in a browser | B2 | M | Organisation rule (D2) | 1; trigger: an exception |
| 15 | GGUF | The llama.cpp ecosystem | B1 | S | Low value: its consumers expect known model architectures [I] | 4; trigger: a request |
| 16 | PTX | NVIDIA without Vulkan; DESIGN.md § 12 keeps it optional | B2 | L | Medium: the driver must accept the PTX version [N7] | 1; last |
| — | AMD HIP direct | — | — | XL | A second runtime for hardware Vulkan already reaches; per-architecture code objects by default, generic AMDGCN SPIR-V optional [N8] | not planned: AMD runs through 2 |
| — | CubeCL, Burn or candle as the backend | — | — | — | Would replace ModelSpec's IR and oracle with theirs [N9] | rejected |
| — | ONNX training graphs | — | — | — | ORT training deprecated [B7] | not planned |

**Order.** 1 with M1's first stories; 2 is M1; 3 runs beside M1 once 1 exists (decision D5); 4
whenever a user needs weights out; 5 after M1's first kernel is green at Kernel level; 6 directly
after 5; 7 after 6. Browser inference comes before browser training because it needs a third of
the kernels and none of the lifecycle handling, so a failure in 5 is about the emitter and the
executor, not about checkpoints. [I]

## 7. How M1 relates

M1 stays what DESIGN.md says: a Vulkan backend with `ash`, SPIR-V generated with `rspirv`,
device discovery and a CPU-versus-Vulkan check (DESIGN.md § 1, § 7). This design changes where
M1's pieces sit and adds three obligations; it adds no M1 story about the browser or the CPU.

| M1 piece | What this design asks | Why |
| --- | --- | --- |
| Lowering to kernels | Build target-neutral Kernel IR and Scheduled IR, each with a reference interpreter, as the input of the SPIR-V emitter, instead of lowering Tensor IR straight to SPIR-V | Every later code target reuses them (§ 4.1, § 4.2) |
| SPIR-V emission | In `modelspec-codegen`, not in `modelspec-runtime` | Emission without a device (§ 4.3) |
| Arithmetic | `NoContraction` on every floating-point instruction in `deterministic` mode; `SignedZeroInfNanPreserve` where the device supports it; no `DenormPreserve` on the reference GPU, which does not support it | [K1], [V, § 1.4], § 5.1 |
| Artifact | Records `NumericContract`; `Backend` becomes a target id | § 4.5, § 12 |
| Conformance | Kernel and step levels against the Scheduled IR interpreter under the emitted policy; lavapipe in CI | § 5.4, § 5.6 |

Dependencies: `rspirv` is current (0.13.0, SPIR-V SDK 1.4.341, released 2026-03-13 [N1]). `ash`
0.38.0 was released on 2024-04-01 and binds Vulkan headers 1.3.281 [N2], while the reference driver
reports Vulkan 1.4.341 [V]. Core compute and `VK_KHR_cooperative_matrix` need nothing newer than
1.3 [I]; whether `ash` 0.38 binds `VK_NV_cooperative_matrix2` I don't know, and it matters only
for M4.

What the browser path takes from M1 and what it does not:

| Shared with M1 | Browser-only | M1-only |
| --- | --- | --- |
| Kernel IR, Scheduled IR, their interpreters; the GPU schedule family (workgroups, workgroup memory tiles, two-stage reductions); the plan; conformance records | WGSL emitter; TypeScript executor; WebGPU limits and features instead of a `DeviceProfile`; OPFS checkpoints | SPIR-V emitter; `ash` executor; `NoContraction` and float controls; cooperative matrix; `f32` atomics (refused under `deterministic` anyway) |

So the WGSL emitter is the second GPU emitter over the same Scheduled IR, which is the cheapest
test that the IR is not secretly SPIR-V-shaped. It starts after M1's first kernel passes its
Kernel-level conformance, not before, so M1 shapes the IR and the browser only checks it. [I]

Decision D1 (§ 10) is the one real fork here. The alternative is `wgpu` as the single native
runtime: one stack for Vulkan, Metal and the browser. Against it for M1: `wgpu`'s cooperative
matrix is experimental and "currently only supports 8x8 f32 matrices" [W21], its `f32` atomics are
native-only [W21], and `no_vulkan.rs` and DESIGN.md's `ash` plan would both change. For it: Metal
comes free. Metal is also reachable from M1's own Vulkan path, through KosmicKrisp (Vulkan 1.4
conformant on Apple silicon with macOS 26) and through MoltenVK on older Macs [N5], so the free
Metal is not a reason to switch. [I]

## 8. Experiment: direct machine code

### 8.1 Question and scope

Can the compiler emit x86-64 machine code for a ModelSpec kernel, with no LLVM and no C compiler
in the path, that is both **bit-identical to the oracle** and **close to the hardware's peak**?
And how much of that peak does exactness cost?

The kernel under test is the core of every M0 forward pass and its loss:

```text
h = relu(matmul(x, w))      x: [M, K], w: [K, N]       fused epilogue
s = sum(h)                  sequential, row-major       the reduction
```

At `M = 64, K = 32, N = 64` this is the first layer of `examples/tiny_mlp.mspec`; the backward pass
of the same model has the transposed products `xᵀ·G` and `G·wᵀ` of the same sizes (DESIGN.md § 4).

Scope: the reference machine (§ 1.4), one ISA level (`x86-64-v3`: AVX2 and FMA), `f32`. AVX-512
and AArch64 are designed for (§ 8.5) but cannot run here: the CPU has no AVX-512 flag [V], and
there is no AArch64 machine.

This does not conflict with DESIGN.md's exclusions as I read them. ADR-006 ("No direct SASS",
"Native instruction compilation remains the vendor driver's responsibility") and the exclusion of
"arbitrary user-authored GPU assembly" sit in the GPU context and concern GPU instruction sets and
user-supplied code; the experiment emits CPU code, generated by the compiler from verified IR, with
every address bounded by static shapes. ADR-006's second sentence could be read more widely, so
section 11 proposes an ADR (ADR-009) that settles it.

### 8.2 Two modes and their ceilings

| Mode | Instructions per product term | Contract | Ceiling per core per cycle |
| --- | --- | --- | --- |
| **exact** | `vmulps` then `vaddps`, 8 lanes | `Sequential`, `Separate`: bit-identical to the interpreter | 16 FLOP |
| **fma** | `vfmadd231ps`, 8 lanes | `Sequential` per element, `Fused`: within the § 5.3 bound | 32 FLOP |

Both ceilings follow from two 256-bit floating-point execution ports. `vfmadd231ps`, `vmulps` and
`vaddps` on `ymm` registers each have a reciprocal throughput of 0.5 cycles on ports 0 and 1 and a
latency of 4 cycles, as measured on Skylake and Coffee Lake [C1]. Comet Lake, the reference CPU, has
no column of its own; that it times like Coffee Lake is inferred [I]. So `fma` is 2 ports × 8 lanes
× 2 FLOP, and `exact` spends one port slot on the multiply and one on the add, which is BLIS's peak
formula with the FMA factor removed [C3]. Exactness therefore costs half the peak by construction,
and the fairest comparison for the exact mode is its own 16 FLOP ceiling.

Peak in GFLOP/s at the documented clocks [I, arithmetic]:

| Clock | Source | exact, 1 core | fma, 1 core | exact, 10 cores | fma, 10 cores |
| --- | --- | --- | --- | --- | --- |
| Base 3.7 GHz | [C2] | 59.2 | 118.4 | 592 | 1 184 |
| All-core turbo 4.8 GHz | [C8] | 76.8 | 153.6 | 768 | 1 536 |
| Maximum single-core turbo 5.3 GHz | [C2] | 84.8 | 169.6 | — | — |

Intel's product page lists no all-core or AVX2 clock [C2]; the 4.8 GHz all-core figure comes from a
review [C8], and no source read gives an AVX2 frequency offset for this part. That is why the
criteria of § 8.7 are stated in FLOP per counted core cycle, which no clock assumption enters
(§ 8.6); GFLOP/s is reported beside it.

### 8.3 Why the exact mode is exact

The interpreter computes `c[i,j] = fl(…fl(fl(p₀ + p₁) + p₂)… + p_(K−1))` with
`p_k = fl(x[i,k] · w[k,j])`, in `f32`, with round-to-nearest-even and no fusion (§ 1.2). A
register-blocked kernel that vectorises over `j` (8 consecutive columns per register) performs, in
each lane, exactly that sequence for its own `(i, j)` when four conditions hold:

1. The accumulator starts from `p₀` (a `vmulps`), not from `+0` followed by an add. Starting from
   `+0` changes the result when every term is `−0.0`: the interpreter returns `−0.0`, the `+0`
   start returns `+0.0`, because `+0 + −0 = +0` under round-to-nearest.
2. Terms are added in ascending `k`. Blocking the `k` axis (BLIS's `kc`) is allowed because the
   partial sum is stored as an `f32` and reloaded, which is exact.
3. No split of the `k` axis across threads; threads partition `i` and `j` only.
4. The floating-point control register keeps round-to-nearest-even with flush-to-zero and
   denormals-are-zero off, as the interpreter runs.

The epilogue is exact too. `relu_where(x, x)` returns `x` for `x > 0` and `+0` otherwise, including
`−0.0` and NaN. `VMAXPS dst, x, zero` with the zero register as the **second** source matches it on
every input: "If the values being compared are both 0.0s (of either sign), the value in the second
operand (source operand) is returned", and "If only one value is a NaN (SNaN or QNaN) for this
instruction, the second operand (source operand), either a NaN or a valid floating-point value, is
written to the result" [C5]. With the operands swapped, `relu(−0.0)` would return `−0.0` and
`relu(NaN)` a NaN. The same reasoning gives `ReluGrad`: `vcmpps` with an ordered greater-than
predicate (false on NaN) and `vandps` with `g`. [I, from [C5] and § 1.2]

The reduction `s = sum(h)` is the hard part. Policy `Sequential` over `M·N` elements is one
dependency chain: every add waits for the previous one, so it runs at one add per add-latency
(4 cycles on this core family [C1]) whatever the vector width. Its cost relative to the matmul,
with the matmul at 80 % of the exact ceiling (12.8 FLOP per cycle):

| Shape (M, K, N) | matmul cycles `2MNK / 12.8` | sequential sum cycles `4·M·N` | sum share of total |
| --- | --- | --- | --- |
| 64, 32, 64 (tiny_mlp layer 1) | 20 480 | 16 384 | 44 % |
| 256, 256, 256 | 2 621 440 | 262 144 | 9 % |
| 1024, 1024, 1024 | 167 772 160 | 4 194 304 | 2.4 % |

[I, arithmetic from [C1]]

At M0 sizes the exact reduction is almost half the kernel. Two remedies are in the experiment:

- **Overlap, still exact.** Emit the kernel row-panel by row-panel in ascending `i` and interleave
  the scalar add chain of finished rows with the multiply work of the next panel. The chain uses one
  port slot every 4 cycles, so an out-of-order core can hide it behind independent work. This keeps
  policy `Sequential` and is variant `E-overlap` (§ 8.6). [H]
- **Change the policy, deterministically.** `Strided { lanes: 8, combine: Sequential }` keeps 8
  partial sums in one register and combines them in a fixed order at the end. It is a different
  policy, so it is compared bit for bit with the oracle running `Strided { 8 }` and with
  `Sequential` within the § 5.3 bound. It is offered under `deterministic`, never as the default.

### 8.4 Approach: templates and an analytical model, search only to check

The kernel follows the BLIS structure [C3]: five loops around a register-blocked microkernel, with
`A` and `B` packed into contiguous panels so the microkernel streams memory linearly. Three ways to
choose its parameters were considered:

| Approach | What it decides | Verdict |
| --- | --- | --- |
| Templates | Microkernel shape `mr × nr`, instruction sequence, prefetch distance | **Used.** One parametric template per ISA family and mode. |
| Analytical model [C4] | Cache blocking `mc`, `kc`, `nc` from cache sizes and associativity, `mr × nr` from register count and latency | **Used.** The inputs are in the CPU profile (§ 1.4), so the choice is deterministic and goldenable. |
| Search (autotuning, superoptimisation) | Anything | **Used only to check the model:** a small grid around the model's choice, run once on the reference machine; if the grid finds more than 10 % better, the model is wrong and that is a finding. |

Register budget, AVX2 (16 `ymm` registers), microkernel `6 × 16` (6 rows, two registers of 8
columns):

| Mode | Accumulators | `B` row | `A` broadcast | Product temporary | Total | Per `k` step |
| --- | --- | --- | --- | --- | --- | --- |
| exact | 12 | 2 | 1 | 1 | 16 | 12 `vmulps` + 12 `vaddps` on 2 ports: 12 cycles; 8 loads on 2 load ports: 4 cycles |
| fma | 12 | 2 | 1 | 0 | 15 | 12 `vfmadd231ps`: 6 cycles; 8 loads: 4 cycles |

Both are bound by the floating-point ports, not by loads, and both have 12 independent accumulator
chains, more than the 8 in flight that 2 ports × 4 cycles of latency require. That is Low et al.'s
register-block condition `mr·nr ≥ N_VEC·L_VFMA·N_VFMA` [C4], here `8 · 4 · 2 = 64` against
`6 · 16 = 96`. The `6 × 16` shape is the one BLIS's Haswell `sgemm` kernel uses, with cache blocks
`mc = 168`, `kc = 256` and `nc = 4080` for `f32` [C3]; those are the starting point the analytical
model must reproduce or explain. [I, from [C1], [C3] and [C4]]

### 8.5 Emission

The Scheduled IR for the kernel is turned into instructions by a Rust emitter in
`modelspec-codegen`, with `iced-x86` [N22] as the encoder and its formatter for the listing golden
(§ 5.5). Three ways to get the bytes executed:

| Option | How | For | Against |
| --- | --- | --- | --- |
| A. JIT | Encode with `iced-x86`, copy into an `mmap` region, switch it from writable to executable (never both), call through a typed function pointer | No toolchain at run time; shapes known only at load time can still be specialised; one `unsafe` call site to isolate and test | W^X handling is ours; profilers see an anonymous region unless we register it |
| B. `dynasm-rs` | Assembly templates in Rust macros, assembled at build time, patched at run time [N21] | Templates read like assembly | The template set is fixed at build time; a second mechanism beside the IR-driven emitter; no AVX-512 and no SVE [N21], so it cannot serve rows 9 and 10 of § 6 |
| C. Ahead-of-time | Emit the listing into a generated Rust file inside `core::arch::global_asm!`, or an object file with the `object` crate, and link it | Ordinary symbols for `perf`; no executable memory at run time | Needs `rustc` or a linker at deploy time; no load-time specialisation |

The recommendation is A for the experiment, with the C listing produced from the same instruction
stream for inspection. AVX-512 (32 `zmm` registers, embedded broadcast, masked tails) and AArch64
(NEON, SVE, SME) are further template families of the same emitter; the Scheduled IR does not
change. [I]

### 8.6 Variants, baselines and measurement

| Id | What runs | Bit-exact expected |
| --- | --- | --- |
| E-exact | JIT `6 × 16` exact kernel, `relu` fused, separate sequential sum | yes |
| E-overlap | E-exact with the sum interleaved into the next row panel | yes |
| E-strided | E-exact with `Strided { 8 }` sum | against `Strided { 8 }` oracle |
| E-fma | JIT `6 × 16` fused kernel | no; within bound |
| B-interp | `evaluate::<f32>` on the Tensor IR program (IR walk and allocation included) | reference |
| B-ijk | Rust loop with iterators in the interpreter's `i, j, k` order, `-C target-cpu=native` | yes |
| B-ikj | Rust loop in `i, k, j` order, accumulator initialised from `k = 0`, so LLVM can vectorise over `j` | yes |
| B-intrin | Generated Rust with `core::arch` intrinsics, same schedule as E-exact and E-fma | yes / no |
| B-mm | `matrixmultiply` 0.3.11 `sgemm` (pure Rust, BLIS-style [N30]) | no |
| B-gemm | `gemm` 0.19.0 [N30] | no |
| B-blas | OpenBLAS 0.3.34 `cblas_sgemm`, `OPENBLAS_NUM_THREADS=1`, through FFI in a bench-only feature | no |

B-intrin is the control the operator's question needs: Rust does not fuse `_mm256_mul_ps` and
`_mm256_add_ps` into one instruction [R1; that the rule covers intrinsics is inferred, I], so the same schedule written as intrinsics isolates what direct emission adds
beyond the schedule itself.

Shapes: the M0 shapes (`8 × 4 × 1`; `64 × 32 × 64` and `64 × 64 × 1` forward, and their transposed
backward products), squares `64` to `4096` in powers of two, each also at `−1` and `+1`, and the
skinny `64 × 4096 × 64`. Exactness (S1) runs every shape; speed (S2 to S5) the shapes it names.

Measurement:

- One thread pinned to one core; median of at least 30 timed repetitions after warm-up; inputs from
  the Philox stream of DESIGN.md § 2.3, so every run sees the same data.
- Cycles are counted, not assumed: user-space `cycles` and `instructions` of the benchmark thread
  through `perf_event_open`, which `perf_event_paranoid = 2` permits for one's own process [V].
  Efficiency is `2·M·N·K / cycles`. Counting core cycles makes the result independent of turbo,
  AVX frequency offsets and the `powersave` governor this machine runs. [I]
- Wall-clock GFLOP/s at the observed frequency is reported beside it, and port-dispatch counters
  confirm that the floating-point ports, not loads, are the bottleneck.
- Every number goes into a benchmark record (target, shape, mode, cycles, instructions, wall time,
  CPU frequency observed, command, date) under `benches/kernels/`, which DESIGN.md § 7 reserves.

### 8.7 Success criteria

The experiment succeeds when all of S1 to S4 hold on the reference machine; each is falsified by
one counter-example.

| Id | Criterion | Falsified by |
| --- | --- | --- |
| S1 | E-exact and E-overlap outputs are bit-identical to `evaluate::<f32>` on every shape, NaN compared as a class; the decoded instruction stream contains no FMA instruction. | Any differing bit, or any FMA opcode |
| S2 | E-exact sustains ≥ 12.8 FLOP per cycle (80 % of its 16 FLOP ceiling [C1]; 80 % from [N17, C10]) on `M = N = K` ∈ {1024, 2048, 4096}, one core. | A median below 12.8 on any of the three |
| S3 | E-fma sustains ≥ 25.6 FLOP per cycle (80 % of 32 [C1, N17, C10]) and ≥ 90 % of B-blas's FLOP per cycle on the same shapes, one core. | Either bound missed on any of the three |
| S4 | On `64 × 32 × 64`, E-overlap's latency for `relu(matmul)` plus the exact sum is no more than B-mm's latency for the matmul alone. | B-mm faster |
| S5 | Informative, not a pass condition: if B-intrin reaches ≥ 95 % of E-exact and E-fma on S2 and S3 shapes, direct emission earns nothing in speed over generated intrinsics; its remaining value is load-time specialisation without `rustc`. | — |

Why 80 %. Two published single-core measurements bracket it. On an Intel Tiger Lake core
(i7-1185G7), Exo's, MKL's and OpenBLAS's `sgemm` all reach "between 80-95% of theoretical peak
FLOPS", 137.6 GFLOP/s, on square matrices [N17]. On Zen 2 a hand-written `6 × 16` AVX2 kernel
reached "~93% of BLAS performance and ~75% of the theoretical performance limit", and the BLAS
behind NumPy reached 80 % of that limit [C10]. A Skylake-client write-up puts MKL at 18 of 32 FLOP
per core per cycle, 56 %, but in a multi-threaded run [C11]. No published `sgemm` share of peak
for Comet Lake was found. So 80 % is the bottom of the band where tuned libraries sit: it asks for
library-grade code, not more. S3's second bound
measures the same kernel against OpenBLAS on this machine, so a miss on the first bound can be read
against what the best available library reaches here; it still counts as a miss. [I]

S4 is the test of the thesis behind a whole-program compiler on a CPU: a kernel specialised to
static shapes and fused with its consumers should beat a general library at M0 sizes even while
paying for exactness. If it does not, the CPU machine-code target is a curiosity and the interpreter
plus a library call is the CPU path. [H]

Budget: two focused weeks after the Kernel IR and Scheduled IR exist. If S2 or S3 fails at the end
of the budget, the record of how far it got is the result and work stops there; an S1 failure is a
bug in the design or the emitter and blocks any speed claim.

## 9. Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Too many targets for one small team | Nothing reaches a conformance record | The order of § 6; a target enters only with its conformance level defined; rows below the line stay research |
| Kernel IR designed too wide before M1 needs it | M1 is late | Kernel IR covers the 14 operations and SGD only; M3 operations extend it when they arrive |
| WGSL's floating-point rules are weaker than the oracle's (§ 2.1) | No bit-exact browser result | The contract says so; the comparator uses computed bounds and lockstep, not hope |
| Browser and driver variance | A bundle that passes on one machine fails on another | Conformance records name the adapter; the bundle names required features and limits and refuses early |
| JIT code is `unsafe` and executes generated bytes | Memory corruption from an emitter bug | Static shapes bound every address; the decoder re-reads every module before it is mapped executable; one call site; W^X |
| Emitters drift apart | Targets disagree for reasons nobody can name | One Scheduled IR, one interpreter per level, per-stage translation validation (§ 4.2) |
| ESS breaking changes to `artifact` | Existing artifacts unreadable | Done before any artifact is serialised outside tests, or behind a format-version bump |
| Exact CPU mode costs half the peak | Users choose speed and lose exactness silently | The mode is in the artifact and its hash; `fast_math` is explicit (DESIGN.md § 3.3) |

## 10. Open decisions

Each is reversible until code depends on it; the recommendation is what this design assumes.

| Id | Decision | A | B | C | Recommendation |
| --- | --- | --- | --- | --- | --- |
| D1 | How M1 and the browser path share GPU code | Separate stacks from one Scheduled IR: `ash` + `rspirv` for Vulkan as DESIGN.md says, a second WGSL emitter, `wgpu` only to check WGSL natively in CI | `wgpu` as the one native runtime (Vulkan through `wgpu`), WGSL or SPIR-V passthrough | `naga` IR as the one shader IR, written out as SPIR-V, WGSL and MSL by `naga`'s backends | **A**: M1 keeps SPIR-V-level control (`NoContraction`, float controls, cooperative matrix) and its plan; the browser costs one emitter, not a runtime change (§ 7) |
| D2 | Browser execution form | Ahead-of-time web bundle, executor in TypeScript (Vue) | As A, plus the Rust compiler compiled to `wasm32` in the tab for a playground | A Rust executor compiled to wasm with `wgpu`'s WebGPU backend, Vue for the UI only | **A**: it is the only option inside the organisation rule that browser code is Vue and TypeScript; B and C put Rust-compiled wasm in the browser and need an explicit exception. Generated WGSL is compiler output loaded as data, which this design reads as outside the rule; if that reading is wrong, the browser path itself needs the exception. |
| D3 | How machine code is emitted | JIT: `iced-x86` encoder, `mmap` with W^X | `dynasm-rs` templates assembled at build time | Ahead-of-time: generated Rust with `global_asm!`, or an object file | **A** for the experiment, with C's listing produced from the same instruction stream for inspection (§ 8.5) |
| D4 | Default mode of the CPU machine-code target | `exact` by default; `fma` only under `fast_math` | `fma` by default, checked against bounds | Both emitted, chosen at load time | **A**: matches DESIGN.md § 3.3, where speed that changes results is an explicit opt-in |
| D5 | When the machine-code experiment runs | In parallel with M1, once the Kernel IR and Scheduled IR from M1's first stories exist | After M1 | Before M1, bringing the Kernel IR up on the CPU first | **A**: it needs no GPU, and it exercises the Scheduled IR on a second target while M1 still shapes it |

## 11. Proposed changes to DESIGN.md

None of these is applied. Each names the section it would change.

| § | Change | Why |
| --- | --- | --- |
| 1 | Add a sentence under the milestone table: Kernel IR, Scheduled IR and their interpreters arrive with M1; the WGSL, browser, export and CPU machine-code tracks follow the order in targets.md § 6 and do not block M1. | The targets need a place in the plan without moving M1. |
| 1 | Keep "direct SASS emission" and "arbitrary user-authored GPU assembly" excluded; state that compiler-generated CPU machine code from verified IR is an experimental target. | § 8 must not read as a breach of the exclusions. |
| 2.1 | Give the later `target` construct its meaning: it selects a `TargetId` and constrains capabilities; it never changes model semantics. | The row already says "Backend configuration, not model semantics". |
| 2.2 | Keep the sequential order as the reference and name it `Sequential`; allow the named deterministic policies of targets.md § 4.4 when the artifact records them. | Lets a GPU or SIMD target be deterministic without matching the CPU order. |
| 3 | Replace the single "target lowering" line of the pipeline with Kernel IR, scheduling, Scheduled IR and emission, and add the graph-export branch (targets.md § 4.1). | Two branch points instead of one. |
| 3.1 | Split Kernel IR into target-neutral Kernel IR and Scheduled IR; give both a reference interpreter. | Translation validation per stage (invariant 7). |
| 3.2 | Replace the illustrative `KernelBackend` with `Target`, `Exporter` and `NumericContract` (targets.md § 4.4). | Pure emission apart from device ownership. |
| 3.3 | Record the numeric contract in the artifact hash beside `fast_math` and `deterministic`. | The contract decides how a result is judged. |
| 4 | Replace "GPU results use documented absolute/relative tolerances" with bounds computed by the shadow interpreter (targets.md § 5.3) and lockstep comparison (§ 5.4). | Tolerances derived, not chosen; long runs checkable. |
| 6 | State that `DeviceRuntime` is the executor contract for every environment, including a browser executor outside Rust held to it by conformance records. | The browser executor is TypeScript by organisation rule. |
| 6.1 | Record the target id and numeric contract with every run. | Reproducibility under a documented configuration. |
| 7 | Add `modelspec-codegen` (emitters, no device dependency) and `web/`; move `spirv/{emit,validate}.rs` from the runtime to codegen; extend the `no_vulkan` guard to codegen. | Emission without a device (CI, bundles, the compiler in wasm). |
| 7.1 | `CompiledArtifact` carries a target id; kernels carry a format; a `cpu-ref` artifact has no Dispatch steps. | Generalises the current Cpu/Vulkan pair. |
| 8 | Add `--target <id>`, `--emit <listing>`, `modelspec bundle --target wgsl-webgpu` and `modelspec export --format onnx\|stablehlo\|safetensors`. | CLI for the new paths. |
| 9 | Generalise gate 5 to every code target against the oracle under its contract; add to gate 6 the `naga` validation of WGSL and the no-FMA check of exact modes; add a gate for conformance records of operator-run targets. | One conformance discipline for all targets. |
| 12 | Replace the "Second target" bullet with a pointer to targets.md § 6, which keeps PTX optional and last. | The question is answered there. |
| 12 | Point the "Precision contract" question ("define permitted reassociation, accumulation types, rounding and overflow") at `NumericContract` (targets.md § 4.4, § 5.1) for `f32`; `f16` combinations stay open until M3. | The contract is this design's answer for `f32`. |
| ADRs | ADR-007 target and executor split; ADR-008 numeric contracts and the parameterised oracle; ADR-009 compiler-generated CPU machine code permitted, user assembly not; ADR-010 the browser path is an ahead-of-time bundle with a TypeScript executor (if decision D2 is taken as recommended). | Record the decisions. |

## 12. Proposed ESS changes

Spec first: these types exist before the code that uses them (AGENTS.md). Each change names its
domain file under `spec/domains/`.

| Domain | Change | Kind |
| --- | --- | --- |
| `artifact` | `Backend { Cpu, Vulkan }` becomes `TargetId`, a newtype of `String` with the naming rule of targets.md § 4.4; `Cpu` maps to `cpu-ref`, `Vulkan` to `spirv-vk13`. | breaking |
| `artifact` | New enum `ModuleFormat { SpirV, Wgsl, X86_64, Aarch64, Wasm }`; `KernelModule.spirv: Bytes` becomes `code: Bytes` plus `format: ModuleFormat`. | breaking |
| `artifact` | New struct `NumericContract` with enums `ReductionPolicyKind { Sequential, Strided, TwoStage, Unordered }` (plus `lanes` and `group` fields), `Contraction { Separate, Fused }`, `Denormals { Preserve, FlushToZero, Unspecified }`, `Repeatability { BitExact, Unspecified }`, and a `specials_preserved: Boolean`; new field `CompiledArtifact.numeric_contract`; invariant: `deterministic` implies the reduction kind is not `Unordered`. | additive |
| `artifact` | `KernelCacheKey` gains `target` and `numeric_contract`; `vendor_id`, `device_id` and `driver_version` become optional, present for GPU targets. | breaking |
| `artifact` | The invariant "backend == Vulkan or no Dispatch" becomes "target is not `cpu-ref` or no Dispatch". | breaking |
| `artifact` | New struct `WebGpuRequirements` (required feature names, required limits by name) and an optional `CompiledArtifact.webgpu` present exactly for `wgsl-webgpu`. | additive |
| `training` | `device_profile_id` is present exactly for a Vulkan target; new optional `cpu_profile_id` for a CPU machine-code target. | additive |
| `device` | `DeviceProfile` gains `denorm_preserve_f32`, `signed_zero_inf_nan_preserve_f32`, `rounding_rte_f32` and `float_atomic_add_f32`. | additive |
| `device` | New entity `CpuProfile`: vendor, family, model, feature names, cache sizes per level, logical and physical cores. | additive |
| `device` | New entity `WebGpuAdapterProfile`: adapter vendor and architecture as `GPUAdapterInfo` reports them, feature names, limits by name; recorded by the browser executor. | additive |
| `conformance` (new) | Entity `ConformanceRecord` (fields of targets.md § 5.4) and enum `ConformanceLevel { Kernel, Step, Lockstep, Run, Repeat }`; entity `BenchmarkRecord` (fields of § 8.6). | new domain |

The breaking changes are cheap only while no serialised artifact exists outside tests; I don't
know whether one does, and that should be checked before the change is scheduled.

## 13. Source register

Every source was read on 2026-10-21. Quotations and the figures taken from each are in
`targets-research-notes.md` under the same key.

### Rust, numerics and model sizes (R, K)

| Key | Source | URL |
| --- | --- | --- |
| R1 | Rust RFC 3514, float semantics | https://rust-lang.github.io/rfcs/3514-float-semantics.html |
| R2 | Rust standard library, `f32::mul_add` | https://doc.rust-lang.org/std/primitive.f32.html#method.mul_add |
| R3 | Ipsen, Zhou, "Probabilistic error analysis for inner products", Corollary 8 | https://ar5iv.arxiv.org/html/1906.10465 |
| R4 | Kaplan et al., "Scaling Laws for Neural Language Models", § 2.1 | https://ar5iv.labs.arxiv.org/html/2001.08361 |
| R5 | GPT-2 small configuration | https://huggingface.co/openai-community/gpt2/raw/main/config.json |
| K1 | Vulkan specification, appendix "SPIR-V Environment", "Precision and Operation of SPIR-V Instructions" and validation rules | https://docs.vulkan.org/spec/latest/appendices/spirvenv.html |

### WebGPU, WGSL, WebAssembly and WebNN (W)

| Key | Source | URL |
| --- | --- | --- |
| W1 | gpuweb wiki, Implementation Status | https://github.com/gpuweb/gpuweb/wiki/Implementation-Status |
| W2 | Chrome for Developers, New in WebGPU 144 | https://developer.chrome.com/blog/new-in-webgpu-144 |
| W3 | Chrome for Developers, New in WebGPU 147–148 | https://developer.chrome.com/blog/new-in-webgpu-147-148 |
| W4 | MDN browser-compat-data, `api/GPU.json` | https://raw.githubusercontent.com/mdn/browser-compat-data/main/api/GPU.json |
| W5 | WebGPU, W3C Candidate Recommendation Draft of 15 September 2026: limits, devices | https://www.w3.org/TR/webgpu/#limits, https://www.w3.org/TR/webgpu/#devices |
| W6a–W6f | WGSL: reassociation and fusion (a, b), differences from IEEE 754 (c, e), floating-point accuracy (d), atomic types (f). Quoted from the editor's draft source, because the TR page was cut off when fetched; the anchors match the TR | https://www.w3.org/TR/WGSL/#reassociation-and-fusion, https://www.w3.org/TR/WGSL/#differences-from-ieee754, https://www.w3.org/TR/WGSL/#floating-point-accuracy, https://www.w3.org/TR/WGSL/#atomic-types, https://raw.githubusercontent.com/gpuweb/gpuweb/main/wgsl/index.bs |
| W6g | WebGPU, synchronization in the programming model | https://www.w3.org/TR/webgpu/#programming-model-synchronization |
| W7 | gpuweb issue 4894, `f32` atomics | https://github.com/gpuweb/gpuweb/issues/4894 |
| W8 | web3dsurvey, `maxStorageBufferBindingSize` and `maxBufferSize` | https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize, https://web3dsurvey.com/webgpu/limits/maxBufferSize |
| W9 | Chrome for Developers, New in WebGPU 120 (`f16`) | https://developer.chrome.com/blog/new-in-webgpu-120 |
| W10 | Chrome for Developers, New in WebGPU 134 (subgroups) | https://developer.chrome.com/blog/new-in-webgpu-134 |
| W11 | Chrome for Developers, New in WebGPU 155–156 | https://developer.chrome.com/blog/new-in-webgpu-155-156 |
| W12 | Chrome for Developers, New in WebGPU 120 (timestamp quantisation) | https://developer.chrome.com/blog/new-in-webgpu-120 |
| W13 | Chrome for Developers, timer throttling in Chrome 88 | https://developer.chrome.com/blog/timer-throttling-in-chrome-88 |
| W14 | Google Chrome Help, Memory Saver | https://support.google.com/chrome/answer/12929150?hl=en |
| W15 | V8 blog, up to 4 GB of memory in WebAssembly | https://v8.dev/blog/4gb-wasm-memory |
| W16 | WebAssembly feature table data; MDN, `SharedArrayBuffer` security requirements | https://raw.githubusercontent.com/WebAssembly/website/main/features.json, https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements |
| W17 | SpiderMonkey blog, "Is Memory64 actually worth using?" | https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html |
| W18 | WebAssembly relaxed SIMD proposal, overview | https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md#relaxed-fused-multiply-add-and-fused-negative-multiply-add |
| W19 | WebNN, W3C Candidate Recommendation Draft of 10 October 2026; Web Machine Learning Working Group charter | https://www.w3.org/TR/webnn/, https://www.w3.org/groups/wg/webmachinelearning/charters/active/ |
| W20 | Chrome Platform Status, WebNN | https://chromestatus.com/feature/5176273954144256 |
| W21 | `wgpu` 30.0.1, `Features` | https://docs.rs/wgpu/latest/wgpu/struct.Features.html |
| W22 | `naga` README | https://github.com/gfx-rs/wgpu/blob/trunk/naga/README.md |
| W23 | Dawn repository | https://dawn.googlesource.com/dawn/ |
| W24 | WebGPU, `GPU` interface | https://www.w3.org/TR/webgpu/#gpu-interface |
| W25 | MDN, Origin private file system | https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system |

### Browser machine-learning stacks (B)

| Key | Source | URL |
| --- | --- | --- |
| B1 | WebLLM paper (arXiv 2412.15803v2) | https://arxiv.org/html/2412.15803v2 |
| B2 | MLC blog, WebLLM engine; WebLLM README | https://blog.mlc.ai/2024/06/13/webllm-a-high-performance-in-browser-llm-inference-engine, https://github.com/mlc-ai/web-llm |
| B3 | WebLLM model configuration, `vram_required_MB` | https://raw.githubusercontent.com/mlc-ai/web-llm/main/src/config.ts |
| B4 | Maczan, WebGPU dispatch overhead and kernel study (arXiv 2604.02344v1) | https://arxiv.org/html/2604.02344v1 |
| B5 | LlamaWeb, llama.cpp's WebGPU back end on 16 devices (arXiv 2605.20706v1) | https://arxiv.org/html/2605.20706v1 |
| B6 | ONNX Runtime blog, on-device training in the browser | https://opensource.microsoft.com/blog/2024/02/06/on-device-training-training-a-model-in-browser |
| B7 | ONNX Runtime 1.20.0 release notes | https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0 |
| B8 | `onnxruntime-web` distribution listings, 1.19.2 and latest | https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/, https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ |
| B9 | ONNX Runtime Web, WebGPU execution provider and overview | https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html, https://onnxruntime.ai/docs/tutorials/web/ |
| B10 | transformers.js v4 announcement; repository | https://huggingface.co/blog/transformersjs-v4, https://github.com/huggingface/transformers.js |
| B11 | TensorFlow.js WebGPU back end README; platform guide; npm registry entry | https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-webgpu, https://www.tensorflow.org/js/guide/platform_environment, https://registry.npmjs.org/@tensorflow/tfjs-backend-webgpu |
| B12 | Burn repository; `mnist-inference-web` example | https://github.com/tracel-ai/burn, https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web |
| B13 | Burn issue 1254 | https://github.com/Tracel-AI/burn/issues/1254 |
| B14 | candle WebAssembly examples; `candle-core` manifest | https://github.com/huggingface/candle/tree/main/candle-wasm-examples, https://raw.githubusercontent.com/huggingface/candle/main/candle-core/Cargo.toml |
| B15 | candle pull request 3379, experimental `wgpu` back end | https://github.com/huggingface/candle/pull/3379 |
| B16 | Piston / Sequence Toy, training a language model in the browser | https://vin.how/blog/train-a-language-model-in-your-browser |
| B17 | TabGPT repository | https://github.com/jbingen/tabgpt |
| B18 | Optimising a WebGPU matmul kernel | https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel |
| B19 | Answer.AI, gpu.cpp | https://answer.ai/posts/2024-07-11--gpu-cpp.html |
| B20 | `webgpu-blas` repository | https://github.com/milhidaka/webgpu-blas |
| B21 | Doppler README 0.5.1; training handbook | https://cdn.jsdelivr.net/npm/doppler-gpu@0.5.1/README.md, https://github.com/clocksmith/doppler/blob/main/docs/training-handbook.md |
| B22 | Apple M1, GPU figures | https://en.wikipedia.org/wiki/Apple_M1 |

### Native GPUs, compilers and formats (N)

| Key | Source | URL |
| --- | --- | --- |
| N1 | crates.io, `rspirv` | https://crates.io/api/v1/crates/rspirv |
| N2 | crates.io, `ash` | https://crates.io/api/v1/crates/ash |
| N3 | `wgpu` release v30.0.1 | https://github.com/gfx-rs/wgpu/releases/tag/v30.0.1 |
| N4 | Phoronix, lavapipe officially Vulkan 1.3 | https://www.phoronix.com/news/Lavapipe-Vulkan-1.3-Official |
| N5 | Khronos, KosmicKrisp achieves Vulkan 1.4 conformance; LunarG on KosmicKrisp and MoltenVK | https://www.khronos.org/news/permalink/kosmickrisp-achieves-vulkan-1.4-conformance-on-apple-silicon, https://www.lunarg.com/?p=4048 |
| N6 | `cudarc` on crates.io and docs.rs, its driver source; NVRTC documentation | https://crates.io/api/v1/crates/cudarc, https://docs.rs/cudarc/latest/cudarc/, https://docs.rs/cudarc/latest/src/cudarc/driver/result.rs.html, https://docs.nvidia.com/cuda/nvrtc/index.html#installation |
| N7 | NVIDIA, CUDA minor version compatibility | https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html#application-considerations-for-minor-version-compatibility |
| N8 | ROCm, HIP support in clang; code portability | https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/clang/html/HIPSupport.html, https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/code-portability.html |
| N9 | CubeCL on crates.io; repository | https://crates.io/api/v1/crates/cubecl, https://github.com/tracel-ai/cubecl |
| N10 | Vulkan reference pages, `VK_KHR_cooperative_matrix` and `VK_NV_cooperative_matrix2` | https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html, https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_matrix2.html |
| N11 | llama.cpp discussions 10879 (Vulkan) and 15013 (CUDA) | https://github.com/ggml-org/llama.cpp/discussions/10879, https://github.com/ggml-org/llama.cpp/discussions/15013 |
| N12 | IREE, deployment configurations | https://iree.dev/guides/deployment-configurations/ |
| N13 | StableHLO compatibility; VHLO dialect | https://openxla.org/stablehlo/compatibility, https://openxla.org/stablehlo/vhlo |
| N14 | Ansor (arXiv 2006.06762); MetaSchedule (arXiv 2205.13603) | https://arxiv.org/abs/2006.06762, https://arxiv.org/abs/2205.13603 |
| N15 | Triton repository | https://github.com/triton-lang/triton |
| N16 | Halide, MIT DSpace record | https://dspace.mit.edu/handle/1721.1/85943 |
| N17 | "Exocompilation for Productive Programming of Hardware Accelerators", § 7.2 | https://cap.csail.mit.edu/sites/default/files/research-pdfs/Exocompilation%20for%20Productive%20Programming%20of%20Hardware%20Accelerators.pdf |
| N18 | Cranelift README; Copy-and-Patch compilation (arXiv 2011.13127), Fig. 22 | https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/README.md, https://arxiv.org/abs/2011.13127 |
| N19 | `inkwell` repository and crates.io; `llvm-sys` OrcV2 | https://github.com/TheDan64/inkwell, https://crates.io/api/v1/crates/inkwell, https://docs.rs/llvm-sys/latest/llvm_sys/orc2/index.html |
| N20 | Rust 1.89.0 announcement; `std::simd` | https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/, https://doc.rust-lang.org/std/simd/index.html |
| N21 | `dynasm-rs` repository | https://github.com/CensoredUsername/dynasm-rs |
| N22 | `iced-x86` README | https://github.com/icedland/iced/blob/master/src/rust/iced-x86/README.md |
| N23 | oneDNN `jit_generator.hpp` (corroborates C14's Xbyak) | https://github.com/uxlfoundation/oneDNN/blob/main/src/cpu/x64/jit_generator.hpp |
| N24 | `core::arch::wasm32`, `f32x4_relaxed_madd` and `f32x4_add` | https://doc.rust-lang.org/core/arch/wasm32/fn.f32x4_relaxed_madd.html, https://doc.rust-lang.org/core/arch/wasm32/fn.f32x4_add.html |
| N25 | ONNX versioning; `onnx.proto` | https://github.com/onnx/onnx/blob/main/docs/Versioning.md, https://github.com/onnx/onnx/blob/main/onnx/onnx.proto |
| N26 | ONNX Runtime 1.20.0 release notes (as B7) | https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0 |
| N27 | crates.io, `ort`; `tract` repository | https://crates.io/api/v1/crates/ort, https://github.com/sonos/tract |
| N28 | GGUF specification | https://github.com/ggml-org/ggml/blob/master/docs/gguf.md |
| N29 | safetensors repository; crates.io | https://github.com/huggingface/safetensors, https://crates.io/api/v1/crates/safetensors |
| N30 | `matrixmultiply` repository; crates.io, `gemm` | https://github.com/bluss/matrixmultiply, https://crates.io/api/v1/crates/gemm |

N31 and N32 re-checked R2 and R1 and are filed under those keys.

### CPU kernels and machine code (C)

| Key | Source | URL |
| --- | --- | --- |
| C1 | uops.info, `VFMADD231PS`, `VMULPS`, `VADDPS`, `VBROADCASTSS` (ymm forms) | https://uops.info/html-instr/VFMADD231PS_YMM_YMM_YMM.html, https://uops.info/html-instr/VMULPS_YMM_YMM_YMM.html, https://uops.info/html-instr/VADDPS_YMM_YMM_YMM.html, https://uops.info/html-instr/VBROADCASTSS_YMM_M32.html |
| C2 | Intel, Core i9-10900K specifications | https://www.intel.com/content/www/us/en/products/sku/199332/intel-core-i910900k-processor-20m-cache-up-to-5-30-ghz/specifications.html |
| C3 | BLIS performance notes; Haswell configuration; Van Zee and van de Geijn, BLIS | https://github.com/flame/blis/blob/master/docs/Performance.md, https://raw.githubusercontent.com/flame/blis/master/config/haswell/bli_cntx_init_haswell.c, https://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf |
| C4 | Low, Igual, Smith, Quintana-Orti, "Analytical Modeling is Enough for High Performance BLIS" | https://www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf |
| C5 | MAXPS instruction reference | https://www.felixcloutier.com/x86/maxps |
| C8 | Guru3D, i9-10900K review, clocks table; hwcooling review | https://www.guru3d.com/review/intel-core-i9-10900k-processor-review/page-2/, https://hwcooling.net/en/intel-core-i9-10900k-the-last-of-the-mohicans |
| C9 | Goto, van de Geijn, "Anatomy of High-Performance Matrix Multiplication" | https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_final.pdf |
| C10 | Algorithmica, matrix multiplication | https://en.algorithmica.org/hpc/algorithms/matmul/ |
| C11 | siboehm.com, fast matrix multiply on the CPU | https://siboehm.com/articles/22/Fast-MMM-on-CPU |
| C12 | Further measurements (Stanford BLAS notes, HazyResearch blocking tutorial, salykova, mmperf) | https://cs.stanford.edu/people/shadjis/blas.html, https://github.com/HazyResearch/blocking-tutorial, https://salykova.github.io/matmul-cpu, https://github.com/mmperf/mmperf |
| C13 | libxsmm repository; BLIS small-matrix performance | https://github.com/libxsmm/libxsmm, https://github.com/flame/blis/blob/master/docs/PerformanceSmall.md |
| C14 | oneDNN CPU dispatcher; bundled assemblers; Xbyak; BRGEMM ukernel; BRGEMM paper | https://uxlfoundation.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html, https://github.com/uxlfoundation/oneDNN/tree/main/third_party, https://github.com/herumi/xbyak, https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_brgemm.html, https://arxiv.org/abs/1906.06440 |
| C15 | AMX; TDPBF16PS; AMX-TF32 removal | https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions, https://www.felixcloutier.com/x86/tdpbf16ps, https://www.phoronix.com/news/Intel-Kills-AMX-TF32 |
| C16 | Alder Lake, Zen 4, Zen 5 | https://en.wikipedia.org/wiki/Alder_Lake, https://en.wikipedia.org/wiki/Zen_4, https://en.wikipedia.org/wiki/Zen_5 |
| C17 | AArch64; Arm SME introduction; Linux SME documentation | https://en.wikipedia.org/wiki/AArch64, https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction, https://docs.kernel.org/arch/arm64/sme.html |
| C18 | Firestorm SIMD timings; Apple M4 SME study (arXiv 2409.18779) | https://dougallj.github.io/applecpu/firestorm-simd.html, https://arxiv.org/html/2409.18779 |
| C19 | Halide autoscheduler, 2019 | https://halide-lang.org/papers/autoscheduler2019.html |
| C20 | Ansor (arXiv 2006.06762) | https://arxiv.org/abs/2006.06762 |
| C21 | egg; Tensat; TASO | https://arxiv.org/abs/2004.03082, https://research.google/pubs/equality-saturation-for-tensor-graph-superoptimization/, https://par.nsf.gov/biblio/10129637 |
| C22 | STOKE, stochastic superoptimisation (arXiv 1211.0557) | https://arxiv.org/abs/1211.0557 |
| C23 | AlphaDev; AlphaEvolve | https://deepmind.google/discover/blog/alphadev-discovers-faster-sorting-algorithms/, https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ |
| C24 | KernelBench; KernelBench-Verified; KernelGenBench | https://arxiv.org/abs/2502.10517, https://www.alphaxiv.org/abs/2607.16241, https://arxiv.org/abs/2607.27231 |

C6 and C7 are not used; the encoders and Rust GEMM crates are under N21, N22 and N30.
