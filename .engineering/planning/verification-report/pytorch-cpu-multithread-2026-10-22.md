---
format: aep.planning-md/3
id: verification-report:pytorch-cpu-multithread-2026-10-22
kind: verification-report
status: draft
title: 'PyTorch multi-thread CPU baseline on MNIST 784-64-10: 314 us/step at 8 threads (contended)'
relations:
- verifies: epic:pytorch-10x-mnist
- serves: vision:modelspec
revision: 1
---
# PyTorch multi-thread CPU baseline, MNIST 784-64-10, 2026-10-22

PyTorch eager on the CPU is fastest at 8 threads: 314 µs/step median. That is the CPU number the
10x target in `epic:pytorch-10x-mnist` is measured against, alongside the 1-thread 526 µs/step of
`verification-report:mnist-benchmark-0-1-1`.

## Setup

- torch 2.14.1+cpu, Intel Core i9-10900K (10 cores, 20 threads), GPU hidden
  (`CUDA_VISIBLE_DEVICES=` and a CPU-only build).
- The benchmark script is unchanged from the 0.1.1 benchmark: same data, same initial weights, same
  order of samples, 1,800 steps of batch 100.
- Threads are pinned with `taskset`. Each series is 10 repeats; median, p10 and p90 are over the
  repeat averages.
- Test accuracy was 0.9088 in every series.
- PyTorch is kept outside the repository; the raw results are in the benchmark directory outside
  the repository (`results/mt-2026-10-22/` and `results/cpu-multithread-2026-10-22.md`).

## Results (µs/step, all contended)

| threads | mode | median | p10 | p90 | 1-min load at start | build slots held |
|---|---|---|---|---|---|---|
| 1 | eager | 451 | 447 | 469 | 6.57 | 2 |
| 2 | eager | 379 | 373 | 403 | 10.81 | 3 |
| 4 | eager | 318 | 313 | 326 | 6.09 | 3 |
| 8 | eager | 314 | 309 | 340 | 6.92 | 3 |
| 20 | eager | 23,367 | 4,722 | 32,793 | 10.98 | 3 |
| 8 | torch.compile (inductor) | 430 | 394 | 643 | 149.76 | 3 |

## Caveats

- Every series ran while other sessions held build slots. A 60-minute wait for a quiet window did
  not find one (load 10.44 at 05:31Z, 22.70 at 05:41Z), so no quiet rerun exists. The figures are
  upper bounds on the quiet times.
- 4 and 8 threads are within noise of each other.
- The 20-thread series shared every logical CPU with 3 builds; it measures contention, not
  scaling.
- The compiled series started at load 149.76, partly the tail of the 20-thread series; it is not
  evidence that compile is slower than eager on a quiet machine.

## What it changes

- 10x over the best CPU figure is at most 31.4 µs/step on the GPU (314 / 10, arithmetic).
