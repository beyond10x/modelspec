---
format: aep.planning-md/3
id: verification-report:mlp-size-sweep-2026-10-23
kind: verification-report
status: draft
title: 'MLP size sweep on the RTX 3090: ModelSpec ahead at batch 100, PyTorch up to 4.3x ahead at batch 8192'
relations:
- reviews: story:mlp-size-sweep-vs-pytorch-cuda
revision: 1
---
# MLP size sweep against PyTorch CUDA on the RTX 3090

One sweep on 2026-10-23, 19:28:11Z to 19:44:17Z, by the coordinating agent, with no warm-up run before it (approval-record:benchmark-wave-scope D14), from wave/benchmark at b2f9553. Each side measures each point 3 times after its own warm-up steps; every number below is the median of those 3 runs.

## Commands

```console
cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep > modelspec.jsonl
python opponents/mlp_sweep/mlp_sweep.py > pytorch.jsonl
cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep -- --crossover modelspec.jsonl pytorch.jsonl
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 100
```

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04 (`nvidia-smi`), clocks not locked |
| CPU, kernel | Intel Core i9-10900K, Linux 6.18.49 |
| PyTorch | 2.14.1+cu130, CUDA 13.0, Python 3.14.7; TF32 off; `torch.compile` mode default |
| ModelSpec | 0.6.0 at b2f9553, `dirty: false`, Vulkan device 10de:2204 |
| other compute processes at start | none (`nvidia-smi --query-compute-apps`) |
| clock samples | 9634, every 100 ms over the whole sweep |

The PyTorch records say `dirty: true`: the tree then held uncommitted planning-store files only (`.engineering/`), no source.

## Per point

FLOP per step is the matmul-only count from the shapes; GFLOP/s is that count over the median µs. The SM clock column is the range and median of the samples inside the 3 timed spans of that point.

| batch | hidden | FLOP/step | side | µs/step (median of 3) | GFLOP/s | SM MHz during runs |
|---|---|---|---|---|---|---|
| 100 | 64 | 20454400 | modelspec-vulkan | 28.058 | 729.0 | 1950–1950 (median 1950, n=176) |
| 100 | 64 | 20454400 | pytorch-eager | 537.472 | 38.1 | 1860–1860 (median 1860, n=215) |
| 100 | 64 | 20454400 | pytorch-compile | 543.529 | 37.6 | 1860–1860 (median 1860, n=164) |
| 100 | 256 | 81817600 | modelspec-vulkan | 46.636 | 1754.4 | 1935–1935 (median 1935, n=183) |
| 100 | 256 | 81817600 | pytorch-eager | 436.056 | 187.6 | 1860–1860 (median 1860, n=170) |
| 100 | 256 | 81817600 | pytorch-compile | 548.911 | 149.1 | 1860–1860 (median 1860, n=204) |
| 100 | 1024 | 327270400 | modelspec-vulkan | 104.330 | 3136.9 | 1830–1875 (median 1860, n=174) |
| 100 | 1024 | 327270400 | pytorch-eager | 440.052 | 743.7 | 1860–1860 (median 1860, n=167) |
| 100 | 1024 | 327270400 | pytorch-compile | 546.276 | 599.1 | 1860–1860 (median 1860, n=163) |
| 100 | 4096 | 1309081600 | modelspec-vulkan | 368.039 | 3556.9 | 1740–1830 (median 1755, n=179) |
| 100 | 4096 | 1309081600 | pytorch-eager | 433.230 | 3021.7 | 1890–1920 (median 1920, n=157) |
| 100 | 4096 | 1309081600 | pytorch-compile | 538.286 | 2431.9 | 1860–1890 (median 1860, n=164) |
| 1000 | 64 | 204544000 | modelspec-vulkan | 110.868 | 1844.9 | 1890–1920 (median 1890, n=181) |
| 1000 | 64 | 204544000 | pytorch-eager | 437.987 | 467.0 | 1860–1860 (median 1860, n=168) |
| 1000 | 64 | 204544000 | pytorch-compile | 520.009 | 393.3 | 1860–1860 (median 1860, n=190) |
| 1000 | 256 | 818176000 | modelspec-vulkan | 227.909 | 3589.9 | 1785–1860 (median 1815, n=178) |
| 1000 | 256 | 818176000 | pytorch-eager | 430.110 | 1902.2 | 1860–1860 (median 1860, n=160) |
| 1000 | 256 | 818176000 | pytorch-compile | 576.433 | 1419.4 | 1860–1860 (median 1860, n=198) |
| 1000 | 1024 | 3272704000 | modelspec-vulkan | 740.638 | 4418.8 | 1665–1725 (median 1680, n=157) |
| 1000 | 1024 | 3272704000 | pytorch-eager | 555.117 | 5895.5 | 1815–1920 (median 1875, n=232) |
| 1000 | 1024 | 3272704000 | pytorch-compile | 676.078 | 4840.7 | 1860–1905 (median 1890, n=177) |
| 1000 | 4096 | 13090816000 | modelspec-vulkan | 3008.156 | 4351.8 | 1545–1905 (median 1575, n=180) |
| 1000 | 4096 | 13090816000 | pytorch-eager | 994.726 | 13160.2 | 1575–1875 (median 1635, n=178) |
| 1000 | 4096 | 13090816000 | pytorch-compile | 994.630 | 13161.5 | 1575–1620 (median 1590, n=184) |
| 8192 | 64 | 1675624448 | modelspec-vulkan | 788.775 | 2124.3 | 1830–1890 (median 1845, n=180) |
| 8192 | 64 | 1675624448 | pytorch-eager | 506.894 | 3305.7 | 1860–1905 (median 1890, n=207) |
| 8192 | 64 | 1675624448 | pytorch-compile | 629.200 | 2663.1 | 1860–1860 (median 1860, n=180) |
| 8192 | 256 | 6702497792 | modelspec-vulkan | 1790.996 | 3742.3 | 1680–1770 (median 1725, n=162) |
| 8192 | 256 | 6702497792 | pytorch-eager | 624.574 | 10731.3 | 1620–1890 (median 1665, n=201) |
| 8192 | 256 | 6702497792 | pytorch-compile | 730.855 | 9170.8 | 1575–1875 (median 1650, n=231) |
| 8192 | 1024 | 26809991168 | modelspec-vulkan | 6147.878 | 4360.9 | 1620–1695 (median 1665, n=174) |
| 8192 | 1024 | 26809991168 | pytorch-eager | 1932.105 | 13876.1 | 1440–1560 (median 1515, n=196) |
| 8192 | 1024 | 26809991168 | pytorch-compile | 1931.610 | 13879.6 | 1455–1530 (median 1500, n=192) |
| 8192 | 4096 | 107239964672 | modelspec-vulkan | 31067.114 | 3451.9 | 1500–1890 (median 1590, n=179) |
| 8192 | 4096 | 107239964672 | pytorch-eager | 7290.461 | 14709.6 | 1440–1515 (median 1470, n=186) |
| 8192 | 4096 | 107239964672 | pytorch-compile | 7261.328 | 14768.6 | 1425–1500 (median 1470, n=188) |

## Crossover (first hidden width more than 10 % slower than the previous width, per side and batch)

```text
{"side":"modelspec-vulkan","batch":100,"verdict":"reached","hidden":256,"pitch_slide":false}
{"side":"modelspec-vulkan","batch":1000,"verdict":"reached","hidden":256,"pitch_slide":true}
{"side":"modelspec-vulkan","batch":8192,"verdict":"reached","hidden":256,"pitch_slide":false}
{"side":"pytorch-eager","batch":100,"verdict":"not reached","hidden":null,"pitch_slide":false}
{"side":"pytorch-eager","batch":1000,"verdict":"reached","hidden":1024,"pitch_slide":true}
{"side":"pytorch-eager","batch":8192,"verdict":"reached","hidden":256,"pitch_slide":false}
{"side":"pytorch-compile","batch":100,"verdict":"not reached","hidden":null,"pitch_slide":false}
{"side":"pytorch-compile","batch":1000,"verdict":"reached","hidden":256,"pitch_slide":true}
{"side":"pytorch-compile","batch":8192,"verdict":"reached","hidden":256,"pitch_slide":false}
```

The pitch-slide point at batch 1000: ModelSpec Vulkan at hidden 256, PyTorch eager at 1024, `torch.compile` at 256.

## Refusals

None: all 36 records have `status: measured`.

## What the numbers show

- ModelSpec is faster than PyTorch eager at every batch-100 point and at batch 1000 up to hidden 256 (28.1 µs against 537.5 µs at 100 × 64).
- PyTorch eager is faster from batch 1000 × 1024 and at every batch-8192 point; at 8192 × 4096 it is 4.26 times faster (7290.5 µs against 31067.1 µs).
- ModelSpec stays between 3.4 and 4.4 TFLOP/s on the larger points; PyTorch reaches 14.8 TFLOP/s. Why ModelSpec stops there is not measured here.
