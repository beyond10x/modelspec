---
format: aep.planning-md/3
id: verification-report:mnist-benchmark-cuda-baseline
kind: verification-report
status: draft
title: PyTorch CUDA baseline of the MNIST benchmark example on the RTX 3090
summary: PyTorch eager CUDA 519 us/step, 192,741 img/s, same as 1 CPU thread; compiled 933 us; 10x gate 1,927,410 img/s
relations:
- reviews: epic:pytorch-10x-mnist
revision: 1
---
# PyTorch CUDA baseline of the MNIST benchmark example on the RTX 3090

Run on 2026-10-21, outside the repository, with the operator's approval of a device run. The
script is the one of `verification-report:mnist-benchmark-0-1-1` (`torch_bench.py --device cuda
--epochs 3 --threads 1 --repeat 3`, with `--compiled` for the second case), in a separate
environment with PyTorch `2.14.1+cu130` (CUDA 13.0), on the same data, initialization and shuffle
order. Settings from the script: `torch.use_deterministic_algorithms(True)`, TF32 off,
`set_float32_matmul_precision("highest")`. The host thread was pinned to CPU 3 with `taskset -c 3`.

## Host state

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04 (`nvidia-smi`) |
| GPU at start | 38 % utilization, 404 MiB used by other processes (`nvidia-smi`) |
| load average | 18.17 at start, 20.15 at end (`/proc/loadavg`); shared workstation |

## Results

Timing as in the 0.1.1 benchmark: the training loop including minibatch gather and one-hot
creation, excluding loading, warmup and compilation, evaluation and IO. 1,800 steps of batch 100.

| case | repeats, seconds | median s | µs per step | images/s |
|---|---|---|---|---|
| ModelSpec 0.1.1, CPU reference (0.1.1 report) | | 35.82 | 19,900 | 5,025 |
| PyTorch eager, 1 CPU thread (0.1.1 report) | | 0.947 | 526 | 190,074 |
| PyTorch eager, CUDA | 1.0208, 0.9339, 0.8940 | 0.9339 | 519 | 192,741 |
| PyTorch `torch.compile`, CUDA | 2.2980, 1.6788, 1.3677 | 1.6788 | 933 | 107,218 |

`torch.compile` spent 9.14 s on warmup and compilation; eager 0.18 s. Every repeat of both cases
reached test accuracy 0.9088 and test MSE 0.02621902557373047.

Final weights against ModelSpec 0.1.1 (`results/modelspec-1`):

| tensor | max absolute difference | relative L2 difference |
|---|---|---|
| W1 | 8.63e-6 | 7.00e-6 |
| W2 | 5.81e-7 | 5.11e-7 |

## Reading

- PyTorch on the RTX 3090 is no faster than on one CPU thread for this example: 519 µs against
  526 µs per step. A step is about 20.45 MFLOP (`epic:pytorch-10x-mnist`). Inferred, not
  profiled: the step time is host-side per-operator launch and dispatch cost, not GPU arithmetic.
- The gate of `epic:pytorch-10x-mnist` is therefore 10 × 192,741 = 1,927,410 images/s, 51.9 µs
  per step, on the RTX 3090.
- `torch.compile` with deterministic algorithms was slower than eager here; the cause is not
  measured.

## Limitations

- One run of three repeats on a loaded shared host; the GPU was in use by another process at
  start. Descriptive ranges, not confidence intervals.
- No profile was taken, so the split between host and device time is inferred.
