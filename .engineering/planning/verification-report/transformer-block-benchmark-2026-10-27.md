---
format: aep.planning-md/3
id: verification-report:transformer-block-benchmark-2026-10-27
kind: verification-report
status: draft
title: 'Transformer block on the RTX 3090: ModelSpec 93.9 ms/step at batch 8 against PyTorch eager 3.9 ms and compile 2.0 ms'
summary: 'Unit 22 timed runs at a123c1a: batch 8/32/128, eager 23.8x/46.5x/53.1x faster, compile 47.6x/97.8x/124.9x; step-0 loss CPU = PyTorch 0.74829835, Vulkan within 1.4e-5'
relations:
- reviews: story:transformer-block-benchmark
revision: 1
---
# Transformer block on the RTX 3090: modelspec against PyTorch eager and torch.compile

Reviews `story:transformer-block-benchmark` (unit 22 of wave bert-1, coordinator decisions D81,
D84). The coordinator ran the commands below on the committed tree at a123c1a (`dirty: false`),
the GPU to itself (`nvidia-smi --query-compute-apps` listed no process at the start), clocks never
locked, through `timed-runs.sh` kept with the unit's scratch. The implementing session's smoke
figures (dirty tree, `--min-seconds 1`) are not results and are not quoted here.

## Commands

```console
cargo bench -p modelspec-runtime --features vulkan --bench transformer_block > modelspec.jsonl
python opponents/transformer_block/transformer_block.py > pytorch.jsonl
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 100 > clocks.csv
modelspec train examples/transformer_block.mspec --backend cpu --steps 0 --checkpoint initial.msckpt --write-batch batch
modelspec train examples/transformer_block.mspec --backend vulkan --steps 0
python opponents/transformer_block/transformer_block.py --loss-check initial.msckpt --batch batch
```

The sampler ran over both timed commands and was stopped after the opponent. The two `train`
lines give the ModelSpec CPU and Vulkan `step 0 loss` lines; the last gives PyTorch's on the same
parameters and batch.

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04, clocks not locked |
| ModelSpec | 0.20.0 at a123c1a5046e6069dc4f341e110d58fad2e02b76, `dirty: false`, Vulkan device 10de:2204 |
| PyTorch | torch 2.14.1+cu130, CUDA 13.0, Python 3.14.7, tf32 off, torch.compile mode default |
| other compute processes at start | none |
| clock samples | 3,127, every 100 ms, 18:40:00 to 18:45:14 UTC |
| timing | ModelSpec 18:40:00 to 18:42:15 UTC; PyTorch 18:42:15 to 18:45:14 UTC |

## Results (median of 3 timed runs of at least 5 s each, after warm-up and calibration)

FLOP per step by `matmul-only`: the matrix products of a step, forward and backward,
`N·(42·H² + 24·H·F + 24·S·H)` with `N = B·128`, `H = 128`, `F = 512`, `S = 128`, that is
339,738,624·B (`crates/modelspec-runtime/benches/transformer_block/block.rs`).

| batch | tokens/step | FLOP/step | modelspec µs/step | modelspec tokens/s | modelspec GFLOP/s | PyTorch eager µs/step | eager tokens/s | eager GFLOP/s | PyTorch compile µs/step | compile tokens/s | compile GFLOP/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1,024 | 2,717,908,992 | 93,868.4 | 10,909 | 29.0 | 3,937.2 | 260,084 | 690.3 | 1,971.9 | 519,307 | 1,378.3 |
| 32 | 4,096 | 10,871,635,968 | 188,352.1 | 21,747 | 57.7 | 4,046.9 | 1,012,141 | 2,686.4 | 1,926.7 | 2,125,860 | 5,642.5 |
| 128 | 16,384 | 43,486,543,872 | 641,794.5 | 25,528 | 67.8 | 12,091.1 | 1,355,042 | 3,596.6 | 5,138.5 | 3,188,470 | 8,462.9 |

Ratios of the medians (PyTorch faster by):

| batch | eager / modelspec | compile / modelspec |
|---|---|---|
| 8 | 23.8× | 47.6× |
| 32 | 46.5× | 97.8× |
| 128 | 53.1× | 124.9× |

The three timed repeats of every point, their timed step counts and the SM clock over the timed
spans are in `benchmarks/runs.yaml` (rows `transformer-block-b<B>-<side>-0.20.0`) and in the raw
records. SM clock over the ModelSpec spans: 1,905 to 1,965 MHz; over the PyTorch spans: 1,740 to
1,935 MHz. No point was refused or marked unmeasured.

Measured, not judged: ModelSpec's step time grows with the batch (93.9 ms at 8, 641.8 ms at 128,
6.8× for 16× the tokens) where PyTorch eager's grows 3.1× and compile's 2.6× over the same
range. Where the time goes is not measured by this report; `train --profile` on the example is
the tool for that, and the epic's measurement rules make this report a figure, not a gate.

## First-step loss

The loss at step 0 of ModelSpec's initial parameters (seed 42) on ModelSpec's synthetic batch of
8 sequences, the batch PyTorch reads from `batch/`. Tolerance of DESIGN.md § 9 against the CPU
value `c`: `|g − c| ≤ 1e-4 + 1e-3·|c|` = 8.48e-4.

| side | step 0 loss | |g − c| | within § 9 of the CPU |
|---|---|---|---|
| ModelSpec CPU | 0.74829835 | (reference) | |
| ModelSpec Vulkan | 0.7482847 | 1.37e-5 | yes |
| PyTorch (`--loss-check`) | 0.74829835 | 0 | yes |

The CPU value equals PyTorch's to every printed digit since fix pass 1 (D84, ADV-TB-5): the CPU
`mse` accumulates in `f64` and rounds once. The adversary's hand run before that fix had the CPU at
0.7482849 against PyTorch's 0.74829835 at B = 8 (within tolerance), and outside tolerance at
B = 128 (0.7482758 against 0.74944544); the one-step parameter comparison added under ADV-TB-6
(`--compare-step`, bound 2e-5 on the update excess) is what shows the two sides compute the same
function, with the figures in `opponents/transformer_block/README.md`.

## Raw records

`benchmarks/raw/2026-10-27-transformer-block/`: `modelspec.jsonl`, `pytorch.jsonl` (one line per
point and side), `clocks.csv` (the sampler), `gpu-procs.txt`, `driver.txt`, `commit.txt`,
`times.txt`, `pytorch-version.txt`, `loss-cpu.txt`, `loss-vulkan.txt`, `loss-pytorch.txt` (the
`step 0 loss` lines), `modelspec.stderr.txt`, `pytorch.stderr.txt` (the venv path replaced by
`<venv>`).
