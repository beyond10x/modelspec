---
format: aep.planning-md/3
id: epic:pytorch-10x-mnist
kind: epic
status: draft
title: 10x PyTorch training throughput on the MNIST benchmark example
summary: same device, same accuracy; gated on the RTX 3090 against PyTorch CUDA eager; 10x on one CPU thread is above the core's ceiling
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- informed_by: verification-report:mnist-benchmark-0-1-1
- informed_by: verification-report:mnist-benchmark-cuda-baseline
revision: 2
---
# 10× PyTorch on the MNIST benchmark example

## Outcome

On the benchmark example of `verification-report:mnist-benchmark-0-1-1` (784 → 64 ReLU → 10, no
bias, f32, MSE, SGD lr 0.1, batch 100, 3 epochs, 1,800 steps), ModelSpec trains at least 10× as
many images per second as PyTorch on the same host and the same device, with the same test
accuracy (0.9088 in the 0.1.1 run) and weights within the tolerance of DESIGN.md § 9.

The operator set this goal on 2026-10-21: "we want to beat pytorch for examples like this with 10x".

## Where 10× is possible

Work per step, from the shapes (no input gradient is needed): forward `2·100·784·64 + 2·100·64·10`,
backward `2·784·100·64 + 2·64·100·10 + 2·100·10·64`, about 20.45 MFLOP. Measured in the 0.1.1 run:

| case | seconds per step | GFLOP/s |
|---|---|---|
| ModelSpec CPU reference | 35.82 / 1,800 = 19.9 ms | 1.03 |
| PyTorch eager, 1 CPU thread | 0.947 / 1,800 = 526 µs | 38.9 |
| 10× PyTorch eager | 52.6 µs | 389 |

The host is an Intel Core i9-10900K (`lscpu`; `avx2` and `fma` in `/proc/cpuinfo`, no `avx512f`)
with an NVIDIA RTX 3090 (`lspci`). Inferred, not measured: one AVX2 core has two FMA ports of 8
f32 lanes, so about 32 FLOP per cycle with FMA and 16 without; at the 5.3 GHz single-core turbo
that is about 170 and 85 GFLOP/s. DESIGN.md § 2 forbids fused multiply-add in the reference, so 389
GFLOP/s on one CPU thread is above that ceiling: **on one CPU thread against PyTorch on one CPU
thread, 10× cannot be reached.** The CPU lane targets parity or better, not 10×.

The 10× lane is the RTX 3090, against PyTorch on the same GPU. Measured
(`verification-report:mnist-benchmark-cuda-baseline`): PyTorch eager on CUDA takes 519 µs per step,
192,741 images/s, no faster than on one CPU thread; `torch.compile` takes 933 µs. The gate is
therefore 1,927,410 images/s, 51.9 µs per step. Inferred, not profiled: at about 20 MFLOP per step
the PyTorch step is host-side launch and dispatch cost, not GPU arithmetic. ModelSpec compiles the whole training run, so it can record every step of an epoch
in one Vulkan command buffer, keep the data set and the weights on the device, and fuse the
elementwise operators into the matmul kernels (DESIGN.md § 0 invariant 2, § 1 row M2).

## Measurement

- Baseline: PyTorch eager on the RTX 3090 (CUDA), same initialization and shuffle order, with
  `torch.compile` reported beside it. The PyTorch side, the data preparation and the plots stay
  outside the repository (AGENTS.md: no Python or PyTorch in code, tests or tooling); the
  repository holds a Rust benchmark target and the figures as verification reports.
- Timing as in the 0.1.1 benchmark: the training loop including minibatch assembly, excluding
  setup, compilation, a 20-step warmup, evaluation and IO; 3 repeats, median.
- The gate is the median ratio against PyTorch eager, CUDA, on the RTX 3090 at ≥ 10. The ratio
  against `torch.compile` is reported, not gated.
- GPU runs are the operator's: every device run is started only after the operator says so.

## Lanes

| lane | work | where it is planned |
|---|---|---|
| CPU reference | row-order `matmul`, the same bits | `story:cpu-matmul-row-order` |
| data | `train --data` over a file, so the benchmark runs through the CLI and not a custom adapter | `story:bert-train-data-binding` |
| GPU execution | Vulkan runtime, SPIR-V kernels, `train --backend vulkan` | `epic:m1-vulkan-execution` |
| whole-run GPU schedule | an epoch in one command buffer, data and weights resident, elementwise fusion, tiled matmul | DESIGN.md § 1 row M2; stories not drafted yet |
| benchmark | a Rust benchmark target in the repository and a verification report per run | not drafted yet |

## Design

ADR-002 in DESIGN.md § 12 reads: "Independence from CUDA APIs is prioritized over immediate peak
throughput." This goal keeps the independence and makes throughput a gated target. Before stories
are written for the whole-run GPU schedule, DESIGN.md gains an ADR that states the 10× target and
the baseline above, with its revision bumped.

## Out of scope

- Comparing different devices (ModelSpec on the GPU against PyTorch on the CPU) as the 10× claim.
- Python, PyTorch or CUDA inside the repository.
- A claim about other models before this example passes.
