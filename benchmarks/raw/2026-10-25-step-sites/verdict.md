# Verdict: the step sites against PyTorch eager, b8192 h4096 (story:step-sites-beat-pytorch)

Goal missed. The goal is the modelspec median at or under the PyTorch eager median in the sweep
harness's back-to-back point run. In the judgement session the medians of 3 runs per side are
6,858.9 µs for modelspec and 6,568.5 µs for PyTorch eager. That is a ratio of 1.044 and a gap of
290.4 µs per step. The reference session before the change measured 7,293.8 against 6,565.0 µs,
a ratio of 1.111 (`reference/`). Both changes keep every bit (`samebits/`).

Device: NVIDIA GeForce RTX 3090, driver 610.57.04, clocks never locked; torch 2.14.1+cu130,
TF32 off. The GPU was checked empty before every run (`judgement/apps-before.txt`).

## The judgement session

`cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep -- --point 8192:4096
--min-seconds 6` (the bench binary of the handed-over tree) alternated with `python
opponents/mlp_sweep/mlp_sweep.py --point 8192:4096 --modes eager`, three rounds, with the sampler
running throughout (`judgement/`).

| round | modelspec median µs | its SM MHz | PyTorch eager median µs | its SM MHz |
|---|---|---|---|---|
| 1 | 7,093.6 | 1,035–1,755 | 6,597.3 | 1,635–1,755 |
| 2 | 6,858.9 | 1,650–1,725 | 6,478.5 | 1,605–1,665 |
| 3 | 6,808.4 | 1,620–1,695 | 6,568.5 | 1,575–1,650 |
| median | 6,858.9 | | 6,568.5 | |

Each SM range covers the samples of the run's three timed spans, each after its first second. The
first timed span of modelspec round 1 started at 1,035 MHz and took 7,540.3 µs per step.

## The figure each candidate reached

| candidate | group | figure | source |
|---|---|---|---|
| the `mse` loss sum recorded beside the `w1` gradient (DESIGN.md 0.45) | the rest | 7,293.8 → 7,050.5 µs per step, medians of 3 back-to-back runs; 107.8 to 291.0 µs off in 4 interleaved pairs | `probe-loss/` |
| `relu_grad` as the epilogue of the matmul producing its adjoint (DESIGN.md 0.46) | small-N and the rest | 433.9 µs for the fused matmul against 194.8 + 610.2 = 805.0 µs for the two dispatches, −371.1 µs, medians of 3 interleaved profiled runs of 1,000 steps | `sites/` |
| unroll 2 for narrow tiles, sites 2 and 8 (follow-up) | small-N | not done; about −85 µs from the earlier diagnosis | `../2026-10-25-blockperf/smalln/` |
| the 13 floats per invocation the `w1`-gradient matmul spills (follow-up) | big matmuls | not done; that matmul is +345.2 µs against cuBLAS | `pairing.md`, `../2026-10-25-blockperf/diagnosis.md` |
| `sgd_update` of `w1` at the bandwidth of `_foreach_add_` | the rest | not done; at most −20.6 µs, its whole difference | `pairing.md` |

The unroll and the SGD figures together give about −106 µs against the 290.4 µs gap. Only the
spill candidate is large enough to close the rest, and it has not been measured.

In the profiled runs of `sites/` the steady step went from 7,959.0 µs at base to 7,667.9 µs.
Profiled runs time every dispatch, and they are compared only with profiled runs.
