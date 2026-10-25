---
format: aep.planning-md/3
id: verification-report:mlp-size-sweep-2026-10-25
kind: verification-report
status: draft
title: 'MLP size sweep on the RTX 3090 from the 0.10.0 build: modelspec / PyTorch eager 1.06 at batch 8192 hidden 4096'
relations:
- reviews: story:mlp-size-sweep-vs-pytorch-cuda
revision: 1
---
# MLP size sweep on the RTX 3090 from the 0.10.0 build

One deterministic sweep on 2026-10-25, 2026-10-25T05:59:02.871Z to 2026-10-25T06:05:24.591Z (modelspec), then PyTorch eager and compile to 2026-10-25T06:15:47.835Z, by the coordinating agent from release/0.10.0 at 5b713ce (clean checkout, `dirty: false`), the GPU to itself, clocks never locked. Coordinator decision D38 judges the batch-8192 goal on this run.

## Commands

```console
cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep > modelspec.jsonl
python opponents/mlp_sweep/mlp_sweep.py > pytorch.jsonl
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 100
cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep -- --crossover modelspec.jsonl pytorch.jsonl
```

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04 (`nvidia-smi`), clocks not locked |
| ModelSpec | 0.10.0 at 5b713ce, `dirty: false`, Vulkan device 10de:2204 |
| PyTorch | 2.14.1+cu130 13.0 3.14.7, tf32 off, torch.compile mode default |
| other compute processes at start | none |
| clock samples | 10029, every 100 ms over the whole sweep |

## Results (median of 3 timed runs of at least 6 s, µs per step)

| batch | hidden | modelspec | PyTorch eager | PyTorch compile | modelspec / eager |
|---|---|---|---|---|---|
| 100 | 64 | 30.7 | 393.4 | 497.5 | 0.078 |
| 100 | 256 | 40.9 | 392.4 | 514.7 | 0.104 |
| 100 | 1024 | 79.3 | 402.2 | 508.6 | 0.197 |
| 100 | 4096 | 214.4 | 395.7 | 487.5 | 0.542 |
| 1000 | 64 | 95.3 | 405.8 | 558.3 | 0.235 |
| 1000 | 256 | 212.2 | 415.6 | 539.3 | 0.511 |
| 1000 | 1024 | 336.7 | 436.6 | 549.6 | 0.771 |
| 1000 | 4096 | 1020.2 | 935.4 | 933.4 | 1.091 |
| 8192 | 64 | 588.9 | 424.2 | 520.6 | 1.388 |
| 8192 | 256 | 1243.1 | 530.9 | 527.4 | 2.342 |
| 8192 | 1024 | 2385.7 | 1774.0 | 1828.7 | 1.345 |
| 8192 | 4096 | 7229.0 | 6841.0 | 6782.0 | 1.057 |

## Crossover

```
{"side":"modelspec-vulkan","batch":100,"verdict":"reached","hidden":256,"pitch_slide":false}
{"side":"modelspec-vulkan","batch":1000,"verdict":"reached","hidden":256,"pitch_slide":true}
{"side":"modelspec-vulkan","batch":8192,"verdict":"reached","hidden":256,"pitch_slide":false}
{"side":"pytorch-eager","batch":100,"verdict":"not reached","hidden":null,"pitch_slide":false}
{"side":"pytorch-eager","batch":1000,"verdict":"reached","hidden":4096,"pitch_slide":true}
{"side":"pytorch-eager","batch":8192,"verdict":"reached","hidden":256,"pitch_slide":false}
{"side":"pytorch-compile","batch":100,"verdict":"not reached","hidden":null,"pitch_slide":false}
{"side":"pytorch-compile","batch":1000,"verdict":"reached","hidden":4096,"pitch_slide":true}
{"side":"pytorch-compile","batch":8192,"verdict":"reached","hidden":1024,"pitch_slide":false}
```

Raw records: `benchmarks/raw/2026-10-25-sweep-0.10.0/` (both JSONL files, the clock samples, the crossover output, the opponent version). Rows in `benchmarks/runs.yaml`: `sweep-b<B>-h<H>-modelspec-vulkan-0.10.0`, `-pytorch-cuda-eager-0.10.0`, `-pytorch-cuda-compile-0.10.0`.
