---
format: aep.planning-md/3
id: verification-report:mnist-benchmark-0-1-1
kind: verification-report
status: draft
title: MNIST benchmark of release 0.1.1 against PyTorch on one CPU thread
summary: same accuracy 0.9088, parity 1.6e-7; PyTorch eager 37.8x faster on 1 thread
relations:
- reviews: epic:m0-semantic-core
revision: 1
---
# MNIST benchmark of release 0.1.1 against PyTorch on one CPU thread

A benchmark ran on 2026-10-21 outside the repository: ModelSpec 0.1.1 (commit
`16d3dd2f576450fcc71b07cfea2d7c2b01d1fb4a`) through a custom Rust dataset adapter over the
library API, against PyTorch `2.14.1+cpu` eager and compiled, on MNIST (`ylecun/mnist` revision
`77f3279092a1c1579b2250db8eafed0ad422088c`). The PyTorch side, the data preparation and the plots
are Python and stay outside the repository (AGENTS.md: no Python or PyTorch in code, tests or
tooling); this report carries the figures.

## Setup

| | |
|---|---|
| host | Intel Core i9-10900K, pinned to CPU 3, 1 compute thread; shared workstation, load average 11–16 |
| model | 784 → 64 ReLU → 10, no bias, 50,816 weights, float32 |
| objective | MSE over all one-hot target elements, SGD lr 0.1 |
| data | 60,000 training and 10,000 test images, batch 100, 3 epochs, 1,800 steps |
| fairness | identical saved initialization and shuffle order |
| timing | training loop including minibatch assembly; excludes setup and compilation, a 20-step warmup, evaluation and IO; 3 timing repeats with one seed |

## Results

| case | median seconds | range | images/s | test accuracy | test MSE |
|---|---|---|---|---|---|
| ModelSpec CPU reference | 35.82 | 34.84–37.26 | 5,024 | 0.9088 | 0.026219031689688563 |
| PyTorch eager | 0.947 | 0.819–1.302 | 190,013 | 0.9088 | 0.02621903335571289 |
| PyTorch compiled | 0.970 | 0.963–1.025 | 185,579 | 0.9088 | 0.02621903335571289 |

PyTorch eager is 37.8× faster. PyTorch compiled spent 11.08 s on warmup and compilation.

Numerical parity, ModelSpec against PyTorch:

| tensor | max absolute error | relative L2 error |
|---|---|---|
| first gradient W1 | 1.49e-8 | 1.32e-7 |
| first gradient W2 | 2.24e-8 | 1.57e-7 |
| first update W1 | 7.45e-9 | 8.38e-9 |
| first update W2 | 2.98e-8 | 9.86e-9 |
| final W1 | 6.71e-8 | 1.30e-7 |
| final W2 | 1.19e-7 | 1.21e-7 |

Repeated runs gave identical weights for both ModelSpec and PyTorch.

## Reading

- Correctness: the CPU reference agrees with PyTorch to 1.2e-7 relative after 1,800 steps and
  reaches the same test accuracy, on real data at 50,816 weights.
- Speed: the CPU reference is the oracle (DESIGN.md § 9), not the fast path; GPU speed is M1.
  The CPU `matmul` (`crates/modelspec-runtime/src/cpu/interpreter.rs:313`) walks `b` by column
  in the inner loop. A row-order loop keeps the ascending-`k` reduction of DESIGN.md § 2 and is
  filed as `story:cpu-matmul-row-order`.
- Data: training on MNIST needed a custom adapter over the library; the CLI has no `--data`.
  `story:bert-train-data-binding` adds `train --data`.

## Limitations, as the benchmark states them

- CPU reference implementation against optimized native PyTorch kernels, not Rust against Python.
- Custom Rust dataset adapter, not the release CLI.
- A small MLP within ModelSpec's operators; no claim to state-of-the-art MNIST accuracy.
- Shared workstation; descriptive ranges, not confidence intervals.
