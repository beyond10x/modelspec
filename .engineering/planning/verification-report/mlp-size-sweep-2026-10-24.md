---
format: aep.planning-md/3
id: verification-report:mlp-size-sweep-2026-10-24
kind: verification-report
status: draft
title: 'MLP size sweep on the RTX 3090 from the 0.9.0 build: ModelSpec ahead at batch 100 and at batch 1000 up to hidden 1024, PyTorch 1.7x ahead at batch 8192 hidden 4096'
relations:
- reviews: story:mlp-size-sweep-vs-pytorch-cuda
revision: 1
---
# MLP size sweep on the RTX 3090 from the 0.9.0 build

One deterministic sweep on 2026-10-24, 16:04:47Z to 16:11:45Z, by the coordinating agent from release/0.9.0 at 974780b (clean checkout, `dirty: false`), after the three performance stories of 0.9.0 (register-blocked matmul tiles, `--fast-math`, matmuls reading a transposed operand in place). The PyTorch side is the 2026-10-23 sweep (verification-report:mlp-size-sweep-2026-10-23); its code and the device did not change. Each point is measured 3 times after its own warm-up steps and calibration to at least 6 s per run; every number below is the median of those 3 runs.

## Commands

```console
cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep -- --min-seconds 6 > modelspec.jsonl
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader,nounits -lms 250
```

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04 (`nvidia-smi`), clocks not locked |
| ModelSpec | 0.9.0 at 974780b, `dirty: false`, Vulkan device 10de:2204 |
| other compute processes at start | none (`nvidia-smi --query-compute-apps`) |
| clock samples | 1799, every 250 ms over the whole sweep |

## Per point

FLOP per step is the matmul-only count from the shapes; GFLOP/s is that count over the median µs. PyTorch eager CUDA is the 2026-10-23 median at the same point; the ratio is ModelSpec's median over PyTorch's (below 1: ModelSpec ahead). The SM clock column is the range and median of the samples inside the 3 timed spans.

| batch | hidden | FLOP/step | µs/step 0.9.0 (median of 3) | GFLOP/s | µs/step 0.6.0 | PyTorch eager µs | ratio | SM MHz during runs |
|---|---|---|---|---|---|---|---|---|
| 100 | 64 | 20454400 | 33.044 | 619.0 | 28.058 | 537.472 | 0.061 | 1305–1665 (median 1560, n=69) |
| 100 | 256 | 81817600 | 46.970 | 1741.9 | 46.636 | 436.056 | 0.108 | 1050–1950 (median 1935, n=80) |
| 100 | 1024 | 327270400 | 102.226 | 3201.4 | 104.330 | 440.052 | 0.232 | 1380–1935 (median 1920, n=84) |
| 100 | 4096 | 1309081600 | 324.264 | 4037.1 | 368.039 | 433.230 | 0.748 | 840–1920 (median 1890, n=77) |
| 1000 | 64 | 204544000 | 106.247 | 1925.2 | 110.868 | 437.987 | 0.243 | 1305–1920 (median 1920, n=83) |
| 1000 | 256 | 818176000 | 223.524 | 3660.4 | 227.909 | 430.110 | 0.520 | 1725–1875 (median 1845, n=87) |
| 1000 | 1024 | 3272704000 | 492.324 | 6647.5 | 740.638 | 555.117 | 0.887 | 1590–1890 (median 1875, n=85) |
| 1000 | 4096 | 13090816000 | 1693.329 | 7730.8 | 3008.156 | 994.726 | 1.702 | 975–1875 (median 1770, n=78) |
| 8192 | 64 | 1675624448 | 649.429 | 2580.1 | 788.775 | 506.894 | 1.281 | 1335–1920 (median 1920, n=86) |
| 8192 | 256 | 6702497792 | 1442.550 | 4646.3 | 1790.996 | 624.574 | 2.310 | 945–1845 (median 1830, n=73) |
| 8192 | 1024 | 26809991168 | 3459.146 | 7750.5 | 6147.878 | 1932.105 | 1.790 | 1350–1890 (median 1800, n=78) |
| 8192 | 4096 | 107239964672 | 12507.182 | 8574.3 | 31067.114 | 7290.461 | 1.716 | 780–1815 (median 1702, n=72) |

## What it says

The headline is deterministic: `--fast-math` figures are in `benchmarks/raw/2026-10-24-transposed/` and the CHANGELOG, never in `runs.yaml`. Rows in `benchmarks/runs.yaml`: `sweep-b<B>-h<H>-modelspec-vulkan-0.9.0`, raw records `benchmarks/raw/2026-10-24-sweep-0.9.0/modelspec.jsonl`, clock samples `clocks.csv`.
