# NVIDIA-native code generation for ModelSpec: what it could buy BERT-tiny training on an RTX 3090

Date: 2026-10-29. Research input for an ADR on an optional vendor-specific backend (DESIGN.md § 12
"Second target", ADR-002, ADR-006). No GPU workload was run for this report. Local facts come from
this repository (`benchmarks/runs.yaml`, `crates/modelspec-runtime/tests/fixtures/rtx3090.device.json`,
`spec/domains/data.yaml`, the planning store), from the coordinator's profile of 2026-10-29, and from
`vulkaninfo` on the same host (driver 610.57.04), which only queries the driver.

Labels: **source** is cited (a URL in § Sources, or a file in this repository). **estimate** is mine,
with the arithmetic. **unknown** means not found and not guessed.

## Summary

| # | Route | Ballpark step time (batch 32 × 128, f32 unless stated) | Label | Confidence |
|---|---|---|---|---|
| 1 | Today: Vulkan/SPIR-V, 593 dispatches, 418 barriers | 15.0 ms profiled, 16.8 ms training clock | source (profile 2026-10-29) | measured |
| 2 | Opponents on this card: PyTorch eager / `torch.compile` default mode, TF32 off | 9.9 / 4.9 ms | source (`benchmarks/runs.yaml`) | measured, loaded host |
| 3 | Portable Vulkan after the three in-flight fixes | ~9 ms | estimate (coordinator) | medium |
| 4 | Portable Vulkan, fully fused, f32, deterministic | 2.5–4.5 ms | estimate (§ 6) | medium-low |
| 5 | Row 4 with cooperative-matrix bf16/f16 matmuls (`fast_math` row, not headline) | 2.0–3.5 ms | estimate (§ 2, § 6) | low |
| 6 | Own PTX through the CUDA driver API, same kernel design, CUDA Graph | 2.0–4.0 ms (0–25 % better than row 4) | estimate (§ 3, § 6) | low-medium |
| 7 | Own PTX, whole step in one persistent kernel with grid-wide sync, f32 | 1.3–2.5 ms | estimate (§ 5, § 6) | low |
| 8 | Rows 6–7 plus SASS-level scheduling | 0–10 % off kernel time | source (§ 4) + estimate | low |
| 9 | Row 7 with bf16 tensor-core matmuls and f32 accumulation (extra row) | 1.0–1.8 ms | estimate (§ 6) | low |
| 10 | Practical floor (FLOP + DRAM + sync): f32 / bf16 tensor cores | ≈ 1.0–1.3 / ≈ 0.6–0.9 ms | estimate (§ 1) | medium |

**Reading.** Getting from 15 ms to about 3 ms is portable work: fewer dispatches, fusion and better
kernels. It is the same kind of work that took PyTorch from 9.9 ms (eager) to 4.9 ms (compile). An
NVIDIA-specific path adds at most another 1.3–2×. Almost all of that comes from removing kernel
boundaries (a persistent kernel with a guaranteed grid barrier) and from smaller per-kernel gaps, not
from tensor cores. For the f32 headline, tensor cores buy nothing on this card. The RTX 3090's TF32
tensor peak (35.6 TFLOP/s) equals its FP32 CUDA-core peak, TF32 is not f32, and the Vulkan driver
exposes no TF32 or f32-operand cooperative-matrix shape (§ 2).

### Measure before the ADR

1. **The per-dispatch gap.** Record N trivial dispatches with a barrier after each in one Vulkan
   command buffer, against N kernels in one CUDA Graph, on the 3090, for N = 150 and 600. Today's
   profile implies about 4.6 µs per dispatch (estimate: (15.0 − 12.3) ms / 593, host wait
   included). For CUDA, the published launch cost is 0.5–1.3 µs per kernel in a graph (§ 3). If the
   Vulkan gap at about 150 dispatches is under about 2 µs, row 6 saves under 0.4 ms. In that case
   speed alone does not justify the ADR.
2. **The real opponent.** Measure `torch.compile(mode="reduce-overhead")` (CUDA Graphs) and
   `mode="max-autotune"` on the same model and data. "Beat everything except the same approach" is
   measured against that figure, and it is unknown today.
3. **A persistent kernel in Vulkan first** (§ 3, § 5). On NVIDIA it works in practice when the grid
   fits co-resident occupancy. Only CUDA's cooperative launch guarantees it. If the Vulkan version
   works, the vendor backend's main remaining advantage is that guarantee.

## 1. Floor

### RTX 3090 peaks

All figures are a source: the GA102 whitepaper, Appendix A, Table 9, "RTX 3090 FE", at the
1,695 MHz boost clock. Sparse figures are in brackets.

| Quantity | RTX 3090 |
|---|---|
| SMs / CUDA cores | 82 / 10,496 |
| Peak FP32, non-tensor | 35.6 TFLOP/s |
| FP16 tensor, FP16 accumulate | 142 (284) TFLOP/s |
| FP16 tensor, FP32 accumulate | 71 (142) TFLOP/s |
| BF16 tensor, FP32 accumulate | 71 (142) TFLOP/s |
| TF32 tensor | 35.6 (71) TFLOP/s |
| Memory bandwidth | 936 GB/s (384-bit GDDR6X, 19.5 Gbps) |
| L2 / L1 + shared total / register file total | 6,144 KB / 10,496 KB (128 KB per SM) / 20,992 KB |

- On GA10x, FP32 runs on both datapaths of each SM partition; the whitepaper calls this "2x FP32
  processing". The 35.6 TFLOP/s figure counts a fused multiply-add as two FLOP. Without contraction
  (a separate multiply and add), the ceiling halves (DESIGN.md ADR-009).
- The runs in `benchmarks/runs.yaml` held a median SM clock of 1,890 MHz, about 11 % above the boost
  clock. The f32 peak at that clock is about 39.7 TFLOP/s (estimate: 10,496 × 2 × 1.89 GHz).

### Work per step

- **FLOP with the full head:** 37,446,746,112, matmul only (source: `benchmarks/runs.yaml`,
  `flop_per_step`). The estimate reproduces it exactly. Per forward pass, the encoder linears are
  2 × (4 × 134.2 + 2 × 536.9) MFLOP = 3.221 GFLOP, the attention products 2 layers × 2 × 134.2 MFLOP
  = 0.537 GFLOP, the MLM transform 0.134 GFLOP and the head 4096 × 128 × 8192 × 2 = 8.590 GFLOP.
  That totals 12.482 GFLOP forward, and × 3 for the training step gives 37.447 GFLOP.
- **FLOP with masked rows** (estimate): the selection rate is 0.15 (source: `spec/domains/data.yaml`).
  The gather takes n = expected count + 8σ, rounded up to a multiple of 64 (source: DESIGN.md § 8):
  4096 × 0.15 = 614.4, σ = √(4096 × 0.15 × 0.85) = 22.85, so 614.4 + 182.8 = 797.2, giving
  **n = 832**. The head and transform on 832 rows cost 832 × 128 × (8192 + 128) × 2 = 1.772 GFLOP
  forward. Step: 3 × (3.221 + 0.537 + 1.772) = **16.6 GFLOP**.
- **Memory traffic:** 1.2 GB per step today (source: coordinator), which takes 1.28 ms at 936 GB/s
  (estimate). A fully fused step moves about 0.23 GB (estimate):
  - AdamW reads p, g, m and v and writes p, m and v: 7 × 1.49 M × 4 B = 42 MB.
  - Saved activations: about 6 × 2 MB + 2 × 8 MB per layer, written once and read once, for
    2 × 56 = 112 MB.
  - Gathered logits, if they are not fused into the cross-entropy: 832 × 8192 × 4 B = 27 MB, written
    and read for 54 MB.
  - The token table and its gradient: about 20 MB.

  That traffic takes about 0.25 ms. Activations of 2 MB fit in the 6 MB L2, so some of it never
  reaches DRAM.

### Floor

- **Compute** (estimate): 16.6 GFLOP / 35.6 TFLOP/s = 0.47 ms. bf16 tensor cores at 71 TFLOP/s:
  0.23 ms. In practice, f32 SGEMM on GA102 peaks at about 60 % of nominal. cuBLAS reaches
  23.2 TFLOP/s at 4092³ on an RTX A6000, the same GA102 die (source: Boehm). So realistic f32
  matmul time is 16.6 / 23 ≈ 0.72 ms at best, and more for these thin shapes (estimate).
- **Latency, the binding constraint at d = 128:** one `[4096,128] × [128,128]` matmul is 134 MFLOP,
  which takes 3.8 µs at peak (estimate). That is the same order as one kernel boundary (1–5 µs,
  § 3). The output also has only 32 tiles of 128 × 128 for 82 SMs, or 128 tiles of 64 × 64
  (1.56 waves). The step is therefore bound by boundaries, wave quantization and dependency chains,
  not by FLOP. With 593 dispatches, 2 µs each is 1.2 ms, as much as the whole compute floor
  (estimate).
- **Practical floor** (estimate): 1.0–1.3 ms in f32, which is matmul at 0.7–1.0 ms plus a fused
  remainder of 0.2–0.3 ms that partly overlaps. With bf16 tensor cores, about 0.6–0.9 ms. The
  theoretical floor is about 0.6 ms (0.47 ms compute plus sync).

## 2. Vulkan, portable but with NVIDIA features

### Observed on this host

These come from `vulkaninfo`, an RTX 3090 with driver 610.57.04 (source: local query):

- `VK_KHR_cooperative_matrix` (rev 2), `VK_NV_cooperative_matrix` (rev 1),
  `VK_NV_cooperative_matrix2` (rev 1), `VK_NV_cooperative_vector` (rev 4),
  `VK_KHR_shader_bfloat16` (`shaderBFloat16CooperativeMatrix = true`), `VK_NV_cuda_kernel_launch`
  (rev 2), `VK_EXT_device_generated_commands`, `VK_NV_shader_subgroup_partitioned` and
  `VK_KHR_shader_float_controls`. The subgroup size is fixed at 32 and `subgroupSizeControl` is
  available.
- **KHR cooperative-matrix shapes** (source: `rtx3090.device.json`; planning store
  `decision-blocker:cooperative-matrix-f32-operands`):

  | Operands → accumulator | Shapes |
  |---|---|
  | f16 × f16 → f16 | 16×16×16, 16×8×16, 16×8×8 |
  | f16 × f16 → f32 | 16×16×16, 16×8×16, 16×8×8 |
  | bf16 × bf16 → f32 | 16×16×16 |
  | s8/u8 → s32/u32 | 16×16×32, 16×8×32 |

  There is **no TF32 and no f32-operand shape**. CUDA's PTX has TF32 `mma` on sm_80 and later
  (§ 3); Vulkan on this driver does not expose it.
- **What `VK_NV_cooperative_matrix2` adds** (source: Khronos proposal): workgroup-scope matrices,
  "flexible dimensions", tensor layouts and views with clamping and padding, Accumulator→A/B
  conversions "with optional transpose", reductions on accumulator matrices (a row-wise max is the
  example) and decode callbacks. It adds no new component type. Which coopmat2 feature bits the
  3090 driver sets is **unknown**, because `vulkaninfo` does not print them. `inspect-device` could
  query `VkPhysicalDeviceCooperativeMatrix2FeaturesNV`.

### Published Vulkan-against-CUDA results on NVIDIA

**llama.cpp on an RTX 3090** (source: llama.cpp discussions #10879 and #15013). The workload is
Llama 2 7B Q4_0 in `llama-bench`. The numbers come from different submitters, commits, hosts and
drivers.

| Test | Vulkan (coopmat2) | CUDA | Vulkan / CUDA |
|---|---|---|---|
| pp512, no flash attention | 4,666 t/s | 5,175 t/s | 0.90 |
| pp512, flash attention | 5,201 t/s | 5,560 t/s | 0.94 |
| tg128, no flash attention | 164 t/s | 158 t/s | 1.04 |
| tg128, flash attention | 172 t/s | 162 t/s | 1.06 |

**NVIDIA's own Vulkan ML work** (source: Bolz, Vulkanised 2025, as reported by Phoronix; Bolz,
Vulkan Developer Conference 2026 slides):

- The 2025 talk said Vulkan on NVIDIA is "very competitive with CUDA", in some cases faster, with
  coopmat2.
- The 2026 slides report ">50% speedup from initial fusion/scheduling work", and the pass that
  defers barriers got a graph section "Down to 12 syncs out of 35 nodes".
- The chart on slide 29, read by eye, shows decode before → after fusion and barrier deferral as
  +18 % (deepseek2, RTX 4070), +26 % (qwen3, 4070), +58 % (qwen3, 5090), +35 % (gpt-oss, 5090) and
  +36 % (deepseek2, 5090).
- Each fusion is worth "+3-5%" on MoE models.

**What this means here** (estimate):

- In a mature, NVIDIA-tuned Vulkan backend, kernel quality is within about 10 % of CUDA for
  GEMM-heavy work, and equal or better for latency-bound decode.
- The large wins NVIDIA reports in Vulkan come from fusion and barrier scheduling, both portable.
- Cooperative matrices give only f16/bf16 operands on this card, so they belong to the `fast_math`
  row. Coopmat2's accumulator reductions and flexible dimensions would make a fused head plus
  cross-entropy kernel easier to write.
- Cooperative vectors target small per-thread MLPs with f16 operands. Their value for d = 128
  training is **unknown**.

## 3. CUDA-level routes without the CUDA toolkit at run time

### Loading PTX through the driver

- `cuModuleLoadData` and `cuModuleLoadDataEx` accept "a NULL-terminated PTX, either as output by
  nvcc or hand-written" (source: CUDA Driver API). The driver's own PTX JIT compiles it.
  `libnvidia-ptxjitcompiler.so` ships with the driver (source: CUDA EULA, Attachment A).
- The JIT cache defaults to `~/.nv/ComputeCache` on Linux, with a default size of 1 GiB on desktop
  and a maximum of 4 GiB. It is controlled by `CUDA_CACHE_PATH`, `CUDA_CACHE_MAXSIZE` and
  `CUDA_CACHE_DISABLE` (source: CUDA Programming Guide, environment variables).
- A ModelSpec PTX target would therefore be: a Rust PTX text emitter beside the `rspirv` emitter,
  and `libcuda.so` loaded at run time. That needs no nvcc, NVRTC or toolkit at build or run time
  (estimate, from the two sources above).
- **Rust:** `cudarc` (Apache-2.0 or MIT) wraps the driver API, NVRTC, cuBLAS, cuDNN and others. Its
  default `dynamic-loading` feature "will not require any libraries to be present at build time".
  It loads a module with `ctx.load_module(ptx)` and supports CUDA 11.4–13.4 (source: cudarc
  README). The README does not mention CUDA Graph support (**unknown**). The driver API itself has
  graphs.

### Ampere PTX features (source: PTX ISA)

| Feature | Minimum target |
|---|---|
| `mma` with `.tf32` operands, shapes `m16n8k4` and `m16n8k8` | sm_80 |
| `mma` with `.bf16`, shapes `m16n8k8` and `m16n8k16` | sm_80 |
| `mma` with `.f16`, shape `m16n8k16` | sm_80 |
| `ldmatrix` | sm_75 |
| `cp.async` (asynchronous global-to-shared copy) | sm_80 |

- `cp.async` and `ldmatrix` have no portable Vulkan equivalent. They pay off in tiled matmuls, and
  at our sizes that is perhaps 10–30 % of matmul kernel time (estimate, unmeasured).
- **Contraction:** "An add instruction with an explicit rounding modifier is treated conservatively
  by the code optimizer", whereas "mul/add sequences with no rounding modifiers may be optimized to
  use fused-multiply-add instructions" (source: PTX ISA). `add.rn`, `mul.rn` and `fma.rn.f32` are
  therefore the PTX form of `NoContraction` plus an explicit `Fma`. The bit-for-bit class of
  DESIGN.md § 5.4 should carry over (estimate; the oracle test decides).

### Launch overhead

| Measurement | Hardware | Result | Label |
|---|---|---|---|
| NVIDIA "Getting started with CUDA Graphs" | V100, CUDA 10.1, 20 kernels per step | 2.9 µs kernels; 3.8 µs per kernel with stream launches; 3.4 µs with a graph; about 400 µs once to instantiate the graph | source |
| Hazy Research | H100, dummy kernel | 2.1 µs per launch on a stream, about 1.3 µs with CUDA Graphs; they model about 5 µs of stall per kernel boundary | source |
| PyTorch with CUDA Graphs | MLPerf | 1.12× on BERT (4,096 GPUs), 1.70× on Mask R-CNN; the graphed backbone went from 31 to 6 ms | source |
| ModelSpec Vulkan | RTX 3090 | about 4.6 µs per dispatch, host wait included | estimate (§ Summary) |

- A pre-recorded Vulkan command buffer already plays the host-side role of a CUDA Graph (ModelSpec
  submits once per step). The GPU-side cost of a dispatch plus barrier in NVIDIA's Vulkan driver is
  **unknown**: no published measurement was found. That makes it the first measurement in
  § Summary.

### Persistent kernels

- `cudaLaunchCooperativeKernel` "ensures that the kernel launch is atomic … if the API call
  succeeds, then the provided number of thread blocks will launch", on compute capability 6.0 and
  later (source: CUDA Programming Guide, cooperative groups). This is what makes a grid-wide barrier
  safe.
- Vulkan makes no such promise. Sorensen et al. show that an inter-workgroup barrier is
  deadlock-free in practice once a runtime occupancy probe caps the number of workgroups. Later work
  tests workgroup progress models, Vulkan included (source: OOPSLA 2016; arXiv 2109.06132).

### Hybrid route

`VK_NV_cuda_kernel_launch` lets applications "directly upload PTX kernels and dispatch the kernels
from Vulkan's command buffer". It is "a provisional extension and must be used with caution". Its
binary cache is invalidated by a new driver or a different GPU architecture (source: Vulkan
refpage). This 3090 driver advertises it.

With it, the Vulkan runtime keeps memory, synchronization and scheduling, and PTX replaces only the
hot kernels on NVIDIA. That is the smallest vendor-specific surface available (estimate).

## 4. Below PTX: SASS assemblers

| Work | Architecture | Published gain | Label |
|---|---|---|---|
| maxas (Scott Gray) | Maxwell | SGEMM "at 98% of the theoretical throughput", "as much as 4.8% faster" than cuBLAS; repository archived in 2023 | source |
| TuringAs (Yan, Wang, Chu, PPoPP 2020) | Volta, Turing (sm_70/75 only) | Winograd convolution up to 2.65× over cuDNN 7.6.1 on RTX 2070 and 2.13× on V100, up to 93 % of peak; algorithm and SASS together | source |
| GAS (Yan, Wang, Chu, PPoPP 2021 poster) | Turing | tensor-core HGEMM 1.23× over the equivalent PTX; 0.98× when it keeps ptxas's instruction order, so the gain is instruction scheduling | source |
| CuAssembler | lists sm_60 through sm_86 | no performance claims; "still in its infancy", control codes set by hand, meant for tweaking compiler-generated SASS | source |
| CuAsmRL (arXiv 2501.08071) | GPU not named in the abstract | up to 26 %, mean 9 %, over specialised CUDA kernels, by re-scheduling the `-O3` SASS | source |

**For ModelSpec** (estimate):

- About 0–10 % off the time of a few hot kernels. With about 1–2 ms of kernel time per step, that
  is at most 0.1–0.2 ms.
- The work is fragile: instruction encodings and control codes are undocumented and
  reverse-engineered, and they change with each architecture (sm_86, sm_89, sm_120).
- It runs into the licence clause in § 7.
- ADR-006 (no direct SASS) should stand.

## 5. What the best hand-written stacks achieve

- **llm.c** (raw CUDA GPT-2 training, built on cuBLAS, cuBLASLt and cuDNN flash attention): "a bit
  faster than PyTorch Nightly (by about 7%)". The hardware is not stated (source: README). At GPT-2
  scale, hand-written CUDA beats a compiled PyTorch only by single digits.
- **FlashAttention-2:** about 2× FlashAttention, 50–73 % of the A100's theoretical FLOP/s, and
  225 TFLOP/s per A100 (72 % MFU) training GPT-style models (source: arXiv 2307.08691).
  - At our sequence length of 128 and head dimension 64, one head's K or V is 128 × 64 × 4 B = 32 KB
    in f32. It fits in one SM's shared memory, so FlashAttention's tiling over the sequence buys
    little (estimate).
  - Our attention matmuls are 1.6 GFLOP per step, which is about 0.05–0.15 ms of work against
    1.7 ms today (estimate).
- **cuBLAS and CUTLASS at M = 4096, N = K = 128, f32 on GA102:** no published figure was found
  (**unknown**).
  - Estimate: 6–12 µs per GEMM, including prologue and epilogue, against 3.8 µs at peak.
  - The large-square reference: cuBLAS SGEMM reaches 23.2 TFLOP/s on an RTX A6000 (GA102), and a
    hand-written warp-tiled kernel reaches 21.8 TFLOP/s, 93.7 % of cuBLAS (source: Boehm).
- **tiny-cuda-nn fully fused MLPs** (NVIDIA): benchmarked on an RTX 3090 at widths 64 and 128,
  exactly our d, against TensorFlow 2.5 with XLA. The README shows a chart without numbers, so the
  factor is **unknown here**. JIT fusion is "a performance boost of 1.5x to 2.5x". This is evidence
  that keeping the whole network in registers and shared memory wins at d = 128 (source; the README
  enables half precision by default).
- **Megakernels:** all published results are inference, not training.
  - Hazy Research (Llama-3.2-1B decode at batch 1): on H100, about 1.5× SGLang and 2.5× vLLM, both
    of which already use CUDA Graphs; on B200, more than 3.5× vLLM (source).
  - MPK (Mirage): 1.0–1.7× over SGLang and vLLM on A100, H100 and B200 (source: arXiv 2512.22219).
  - Cohere: 1.25–1.41× end-to-end decode on H100. The post says per-operation kernels already work
    well for training and prefill, "where the workload is large compute-bound GEMMs" (source).
  - Our BERT-tiny step, hundreds of 5–20 µs kernels, is closer to the decode regime than to
    large-model training, so the megakernel argument applies (estimate).
- **Triton:** its tutorial matmul "achieves performance on par with cuBLAS or rocBLAS" (fp16)
  (source). `torch.compile` (Inductor) emits Triton kernels and calls cuBLAS for matmuls, so the
  4.9 ms measured figure already includes Triton-level fusion.
- **Apache TVM:** not researched (**unknown**).

## 6. Our size: what matters at d = 128, and the ranking

### Per-kernel budget

The "today" column comes from the coordinator's profile of 2026-10-29. The other two columns are
estimates.

| Item | Today | Fully optimized | Arithmetic |
|---|---|---|---|
| EmbeddingGrad | 2.3 ms | 0.02–0.05 ms | segment sum over sorted ids: read 2 MB, write 4 MB |
| LayerNormGradGamma | 1.5 | 0.05–0.15 | 7 layer norms × 2 column sums over 4096 × 128 (2 MB each) |
| MatMul (encoder, 10.1 GFLOP) | 1.3 | 0.5–0.8 | 10.1 GFLOP at 12–20 TFLOP/s |
| SumRows (bias gradients) | 0.7 | 0.02–0.1 | folded into column sums or matmul epilogues |
| Attention, 4 kernels | 1.71 | 0.1–0.3 | 1.6 GFLOP, one workgroup per head and tile |
| MLM head + cross-entropy (5.3 GFLOP, 832 rows) | ~2.9 | 0.35–0.6 | 5.3 GFLOP at 10–15 TFLOP/s; cross-entropy in the epilogue |
| Small elementwise, AdamW | ~1.5 | 0.2–0.4 | about 0.23 GB at 600–900 GB/s |
| **Kernel sum** | 12.3 | **1.2–2.4** | |
| Gaps: Vulkan, 100–150 dispatches | 2.7 (593 dispatches) | 0.3–1.0 | 2–7 µs each (**unknown**, measure) |
| Gaps: CUDA Graph, 100–150 kernels | n/a | 0.15–0.5 | 1–3 µs each (§ 3) |
| Gaps: one persistent kernel | n/a | 0.1–0.3 | about 100–150 grid barriers at 1–2 µs (**unknown** on sm_86) |

The ideal kernel sum is reached only partly in practice; I assume a factor of 1.3–1.5. That gives
row 4 of the summary as 1.6–3.4 ms of kernels plus 0.3–1.0 ms of gaps, about 2.5–4.5 ms. Row 6 takes
kernels at 0.85–1.0× of row 4's (`cp.async`, `ldmatrix`, compiler differences of about 10 %, § 2)
plus graph gaps, about 2–4 ms. Row 7 also keeps intermediates in L2 and shared memory, about
1.3–2.5 ms.

### What matters at d = 128

- **Matters:** the number of kernel boundaries, fusion (above all the head with cross-entropy, the
  bias and LayerNorm gradients folded into epilogues, and AdamW as one pass), persistent execution,
  wave quantization of thin outputs, and parallel deterministic reductions over 4,096 rows.
- **Matters only at large d:** tensor-core throughput, FlashAttention's sequence tiling, split-K and
  SASS scheduling. At d = 128 the GEMMs are too small to reach the tensor-core regime, and the
  constraint is latency, not FLOP.

### Ranked routes

| Rank | Route | Expected step | Effort | Portability cost | Determinism cost | Confidence |
|---|---|---|---|---|---|---|
| 1 | Portable fusion and fewer dispatches in Vulkan (the M2 roadmap) | 2.5–4.5 ms | high, already planned | none | none: same ordered sums | medium-low |
| 2 | Persistent kernel in Vulkan, grid capped by occupancy | 2–3.5 ms | high | relies on forward progress the spec does not guarantee; needs an occupancy probe, a watchdog and a fallback | none if the barrier order is fixed | low |
| 3 | Optional PTX target, driver API and CUDA Graph | 2–4 ms | very high: a second emitter, runtime, allocator and profiler path | NVIDIA only; CI needs an NVIDIA GPU | none on the SIMT f32 path (`.rn`, `fma.rn`) | low-medium |
| 4 | PTX persistent megakernel with cooperative launch | 1.3–2.5 ms | very high | NVIDIA only | none on the SIMT path | low |
| 5 | Hybrid: PTX hot kernels through `VK_NV_cuda_kernel_launch` | as rank 3 for the replaced kernels | medium-high | NVIDIA only, and the extension is provisional | none on the SIMT path | low |
| 6 | Cooperative-matrix bf16/f16 in Vulkan (`fast_math` row) | 0.3–0.7 ms off ranks 1–2 | medium | KHR is cross-vendor; NV2 is NVIDIA only | loses bit-for-bit (tensor-core accumulation, § 7) | medium-low |
| 7 | TF32 `mma` through PTX | about 0 on GA102: TF32 peak equals FP32 peak | medium | NVIDIA only | not f32 (10-bit mantissa), loses bit-for-bit | high that it does not pay |
| 8 | SASS scheduling | 0–10 % of kernel time, ≤ 0.2 ms | very high, per architecture | per NVIDIA architecture | none intrinsically | low; not recommended |

## 7. Risks

### Licensing

The CUDA EULA says (source: CUDA EULA):

- § 1.2 (1): "You may not reverse engineer, decompile or disassemble" the SDK. § 1.2 (8) extends
  this to SDK output, for porting it to non-NVIDIA platforms.
- § 2.1: the SDK is licensed "only for use in systems with NVIDIA GPUs".
- § 2.6, Attachment A: lists `libcuda.so` ("NVIDIA CUDA Driver Libraries"), NVRTC, nvJitLink and
  `libnvidia-ptxjitcompiler.so` as distributable. It does not list ptxas.

The open questions are legal, and code reading cannot answer them:

- Emitting PTX from the public ISA document and loading it through the user's installed driver uses
  no SDK component at build time. Whether the driver's own licence constrains that is **unclear;
  ask counsel**.
- A SASS assembler is built from reverse-engineered encodings, typically read from `nvdisasm`
  output. Whether that falls under § 1.2 is also **unclear; ask counsel**.

### Driver JIT and caches

- The first run JIT-compiles every PTX module. The cost for about 100–600 kernels is **unknown**:
  measure it.
- The cache is 1 GiB by default and evicts older entries (source: CUDA environment variables).
  Today the Vulkan pipeline cache plays the same role.
- The binary cache of `VK_NV_cuda_kernel_launch` is invalidated by a new driver or a different
  architecture (source).

### Determinism with tensor cores

Fasi, Higham, Mikaitis and Pranesh tested V100, T4 and A100 (source: PeerJ CS 2021):

- binary16 products are exact.
- Additions use "binary32 arithmetic with round-towards-zero".
- "Only the final result … is normalized; the partial sums are not".
- Summands are "accumulated starting with the largest in absolute value".
- On A100, TF32 and bf16 behave like the T4's binary16/binary32 mode.

Consequences:

- Tensor-core results are deterministic from run to run on one device, but they are neither
  round-to-nearest nor in ascending order. They are never bit-for-bit against the CPU oracle unless
  the oracle models the tensor core. Khattak and Mikaitis (arXiv 2512.07004) publish such models
  for several GPUs; whether sm_86 is covered is **unknown**, as the paper was not read.
- cuBLAS guarantees bit-wise reproducibility only for the same toolkit version, architecture and SM
  count. Multiple streams with a shared workspace or atomics mode break it (source: cuBLAS
  documentation). A ModelSpec-owned PTX path would not depend on cuBLAS.

### Maintenance

- A second emitter and runtime (memory, synchronization, profiling, device discovery) doubles the
  backend surface. Either every kernel change lands twice, or the PTX target covers a subset of ops
  with a per-op fallback.
- Tuning is per architecture: sm_86, sm_89 and sm_120 differ.
- CI needs an NVIDIA GPU.
- A persistent kernel in Vulkan relies on behaviour the spec does not promise, and a driver update
  could turn it into a hang.
- `VK_NV_cuda_kernel_launch` is provisional.

### Evidence gaps

None of the step-time ranges in this report has been measured. The two measurements in § Summary
decide whether the vendor route is worth its cost, so they come before the ADR.

## Sources

1. NVIDIA Ampere GA102 GPU Architecture whitepaper v2. Appendix A, Table 9 (RTX 3090 peaks, memory,
   caches), and "2x FP32 Processing". <https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf>
2. Khronos, `VK_NV_cooperative_matrix2` proposal (feature list). <https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_matrix2.html>
3. Khronos, `VK_NV_cuda_kernel_launch` reference page (PTX dispatch from Vulkan command buffers;
   provisional). <https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cuda_kernel_launch.html>
4. J. Bolz, `vk_cooperative_matrix_perf` (cooperative-matrix benchmark sample; Turing or newer; no
   published figures). <https://github.com/jeffbolznv/vk_cooperative_matrix_perf>
5. llama.cpp, "Performance of llama.cpp with Vulkan" (RTX 3090 coopmat2 pp512/tg128, Llama 2 7B
   Q4_0). <https://github.com/ggml-org/llama.cpp/discussions/10879>
6. llama.cpp, "Performance of llama.cpp on Nvidia CUDA" (RTX 3090 CUDA pp512/tg128, same model).
   <https://github.com/ggml-org/llama.cpp/discussions/15013>
7. Phoronix, "NVIDIA Is Finding Great Success With Vulkan Machine Learning - Competitive With CUDA"
   (report of the Vulkanised 2025 talk). <https://www.phoronix.com/news/NVIDIA-Vulkan-AI-ML-Success>
8. J. Bolz, "Vulkan Machine Learning in ggml/llama.cpp", Vulkan Developer Conference 2026 slides
   (fusion and barrier deferral before and after). <https://vulkan.org/user/pages/09.events/vulkanised-2026/0930-Jeff-Bolz-NVIDIA.pptx.pdf>
9. NVIDIA CUDA Driver API, Module Management (`cuModuleLoadData`, which accepts PTX). <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html>
10. NVIDIA CUDA Programming Guide, CUDA Environment Variables (JIT cache path and size). <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html>
11. NVIDIA CUDA Programming Guide, Cooperative Groups (cooperative launch guarantee). <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html>
12. NVIDIA PTX ISA (`mma` target notes, `ldmatrix`, `cp.async`, rounding modifiers and contraction).
    <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
13. `cudarc` README (driver API wrapper, dynamic loading, licence). <https://github.com/coreylowman/cudarc>
14. NVIDIA blog, "Getting Started with CUDA Graphs" (V100 launch overhead with and without graphs).
    <https://developer.nvidia.com/blog/cuda-graphs/>
15. PyTorch blog, "Accelerating PyTorch with CUDA Graphs" (MLPerf BERT 1.12×, Mask R-CNN 1.70×).
    <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>
16. Hazy Research, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (launch
    cost, megakernel speedups). <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>
17. Mirage Persistent Kernel / MPK, arXiv 2512.22219 (megakernel compiler, 1.0–1.7×). <https://arxiv.org/abs/2512.22219>
18. Cohere, "Megakernels" blog (decode megakernel, 1.25–1.41×; regime argument). <https://cohere.com/blog/megakernels>
19. A. Karpathy, llm.c README (about 7 % faster than PyTorch Nightly). <https://github.com/karpathy/llm.c>
20. T. Dao, FlashAttention-2, arXiv 2307.08691 (2×, 50–73 % of A100 peak, 225 TFLOP/s). <https://arxiv.org/abs/2307.08691>
21. S. Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" (RTX A6000, cuBLAS
    23.2 TFLOP/s f32). <https://siboehm.com/articles/22/CUDA-MMM>
22. NVIDIA tiny-cuda-nn README (fully fused MLPs, RTX 3090, widths 64/128). <https://github.com/NVlabs/tiny-cuda-nn>
23. Triton matrix multiplication tutorial ("on par with cuBLAS or rocBLAS"). <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
24. S. Gray, maxas wiki (98 % of peak, up to 4.8 % faster than cuBLAS). <https://github.com/NervanaSystems/maxas/wiki/Introduction>
25. D. Yan, W. Wang, X. Chu, "Optimizing Batched Winograd Convolution on GPUs", PPoPP 2020
    (TuringAs). <https://home.cse.ust.hk/~weiwa/papers/yan-ppopp20.pdf> ; assembler: <https://github.com/HKBU-HPML/turingas>
26. D. Yan, W. Wang, X. Chu, "POSTER: Simplifying Low-Level GPU Programming with GAS", PPoPP 2021
    (1.23× over PTX, 0.98× with ptxas order). <https://cse.hkust.edu.hk/~weiwa/papers/gas-ppopp21.pdf>
27. CuAssembler README (unofficial SASS assembler, sm_86 listed). <https://github.com/cloudcores/CuAssembler>
28. CuAsmRL, arXiv 2501.08071 (up to 26 %, mean 9 % by SASS re-scheduling). <https://arxiv.org/abs/2501.08071>
29. NVIDIA CUDA EULA (§ 1.2, § 2.1, § 2.6 Attachment A). <https://docs.nvidia.com/cuda/eula/index.html>
30. M. Fasi, N. Higham, M. Mikaitis, S. Pranesh, "Numerical behavior of NVIDIA tensor cores", PeerJ
    Computer Science 7:e330, 2021. <https://peerj.com/articles/cs-330/>
31. F. Khattak, M. Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004 (not read;
    cited for existence only). <https://arxiv.org/abs/2512.07004>
32. NVIDIA cuBLAS documentation, Results Reproducibility. <https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility>
33. T. Sorensen, A. Donaldson et al., "Portable Inter-workgroup Barrier Synchronisation for GPUs",
    OOPSLA 2016. <https://www.doc.ic.ac.uk/~afd/papers/2016/OOPSLA.pdf> ; "Specifying and Testing
    GPU Workgroup Progress Models", arXiv 2109.06132. <https://arxiv.org/abs/2109.06132>
