# Step sites paired with PyTorch eager, b8192 h4096 (story:step-sites-beat-pytorch)

Device: NVIDIA GeForce RTX 3090, driver 610.57.04, clocks never locked; torch 2.14.1+cu130, TF32
off. Tree: 2d83128 with the opponent's `--profile` flag and nothing else (`modelspec` built from
it; every site runs the plan for the device). Source: the sweep's own point source at batch 8192,
hidden 4096 (`pairing/source.mspec`, the text `benches/mlp_sweep/sweep.rs` formats), 1,000 steps.

One session, 2026-10-25 07:36–07:39 UTC, three interleaved rounds, each
`modelspec train source.mspec --backend vulkan --profile` then
`python opponents/mlp_sweep/mlp_sweep.py --point 8192:4096 --modes eager --min-seconds 6 --profile`.
`nvidia-smi --query-compute-apps` was empty before every run (`pairing/apps-before.txt`); the SM
clock, memory clock, temperature and power every 100 ms are in `pairing/clocks.csv`. modelspec
figures are the `mean_us` of each `profile dispatch step` line (steady state after the 20-step
warm-up); PyTorch figures are each kernel's CUDA time over the three timed runs divided by their
steps (`profile kernel ... us_per_step`). Every figure below is the median of the three rounds.
`pairing/kernel-map.txt` names the `aten` op and input shapes each PyTorch kernel runs for.

## Pairing

| site | modelspec dispatch | µs (r1 / r2 / r3) | median | PyTorch kernel(s) doing the same work | median µs | difference |
|---|---|---|---|---|---|---|
| 0 | matmul `[8192,784] x [784,4096]`, `relu` epilogue writing the pre-activation | 2,667.5 / 2,712.3 / 2,768.6 | 2,712.3 | `ampere_sgemm_128x64_tn` (`mm [8192,784]x[784,4096]`) 2,572.0 + `clamp_min` (the `relu`) 344.1 | 2,916.1 | −203.8 |
| 12 | matmul `[784,8192]ᵀ x [8192,4096]` (gradient of `w1`) | 2,846.2 / 2,884.3 / 2,936.1 | 2,884.3 | `cutlass_80_simt_sgemm_128x256_8x4_nt` (`mm [4096,8192]x[8192,784]`) 2,538.1 + its memset 0.9 | 2,539.1 | +345.2 |
| 2 | matmul `[8192,4096] x [4096,10]` (prediction) | 283.8 / 282.8 / 273.8 | 282.8 | `ampere_sgemm_32x32_sliced1x4_tn` 226.6 + its memset 0.9 | 227.6 | +55.2 |
| 7 | matmul `[8192,10] x [10,4096]ᵀ` (adjoint of the hidden activation) | 168.4 / 167.7 / 169.0 | 168.4 | `ampere_sgemm_128x32_nn` (`mm [8192,10]x[10,4096]`) | 177.5 | −9.1 |
| 8 | matmul `[4096,8192]ᵀ x [8192,10]` (gradient of `w2`) | 321.7 / 316.8 / 315.6 | 316.8 | `cutlass_80_simt_sgemm_128x32_8x5_nt` (`mm [10,8192]x[8192,4096]`) 183.9 + `splitKreduce_kernel` 9.7 | 193.5 | +123.3 |
| 4 | `loss_sum Mse`, one workgroup, invocation 0 adds 81,920 terms in row-major order | 280.7 / 287.1 / 289.9 | 287.1 | `mse_kernel_cuda` 2.7 + `reduce_kernel<MeanOps>` 11.9 | 14.6 | +272.5 |
| 5 | elementwise `Sub>Mul@0` (adjoint of the prediction) | 3.3 / 3.0 / 3.3 | 3.3 | `mse_backward_cuda_kernel` 2.7 + `FillFunctor` (2 calls) 2.6 | 5.2 | −2.0 |
| 10 | elementwise `ReluGrad` `[8192,4096]` | 533.1 / 519.4 / 524.6 | 524.6 | `threshold_backward` (`BinaryFunctor ... threshold_kernel_impl`) | 505.8 | +18.8 |
| 14, 15 | `sgd_update` of `w1` and of `w2` | 78.0 + 3.3 / 70.3 + 3.0 / 56.9 + 3.3 | 70.3 + 3.3 | `multi_tensor_apply_kernel` (`_foreach_add_`, both parameters) | 53.0 | +20.6 |

The two memsets per step (1.9 µs, one before each of the two `mm` calls the map names) are split
between sites 2 and 12. Sums of medians: modelspec 7,253.1 µs (the median steady step is
7,254.6: 7,190.4 / 7,254.6 / 7,350.3); PyTorch 6,632.4 µs (the median summed device time is
6,631.4: 6,631.4 / 6,612.5 / 6,729.4; its timed runs under the profiler 6,644.0 / 6,661.7 /
6,746.6 µs per step). Session gap: 621 µs per step.

## Group sums

| group | sites | modelspec µs | PyTorch µs | difference |
|---|---|---|---|---|
| big matmuls | 0, 12 | 5,596.6 | 5,455.1 | +141.4 |
| small-N matmuls | 2, 7, 8 | 768.0 | 598.6 | +169.4 |
| the rest | 4, 5, 10, 14, 15 | 888.6 | 578.6 | +309.9 |

The group with the largest summed difference is **the rest**, and its difference is site 4, the
`mse` loss sum: +272.5 of +309.9.

The `relu` of the forward matmul is paired with site 0, because site 0 computes it in its
epilogue. Counted with the rest instead, the big matmuls would be +485.5 and the rest −34.2.

**The largest group's difference is under the gap.** It is +309.9 µs against 621 µs in this
session, 388 µs on the release sweep (`../2026-10-25-sweep-0.10.0/`), and 729 µs on the
back-to-back reference run below. No single group's difference covers the gap. So the pairing
goes to the coordinator before the change, as coordinator decision D49 (2026-10-25,
`.engineering/waves/perf-1.md`) asks.

## Candidates and the figure each reached

| candidate | group | bits | figure | source |
|---|---|---|---|---|
| the loss sum recorded beside the gradient of `w1` (site 12), with no barrier between them, in place of its own barrier interval after the prediction | the rest | same: `train --backend vulkan` stdout sha256 `ffc7cfdd…` on the 220-step source for base and probe | −107.8, −183.9, −166.7, −291.0 µs per step in 4 interleaved bench pairs; −243.3 in the reference session (7,293.8 → 7,050.5 µs) | `probe-loss/ab/`, `probe-loss/ab2/`, `reference/`; patch `probe-loss/probe-loss-overlap.patch` (unapplied) |
| the same, with the loss sum recorded first in that interval | the rest | same (sha256 as above) | +7.0, −149.0 µs | `probe-loss/ab2/r*-probe-lead.*`; `probe-loss/probe-loss-lead-env.patch` |
| `ReluGrad` computed in the epilogue of site 7: the matmul also reads the pre-activation, writes the masked adjoint, and the adjoint of the hidden activation gets no buffer | the rest (and site 7) | same by construction: a compare and select, no rounding | not measured; inferred −350 µs: site 10's 524.6 µs goes away, and site 7 reads 134 MB more at the ~800 GB/s sites 7 and 10 reach | — |
| unroll 2 for narrow tiles (output width under 16), sites 2 and 8 | small-N | same: the order of each element is unchanged | from the earlier diagnosis: site 2 282.5 → 229.6, site 10 (now 8) 305.4 → 272.9, about −85 µs | `../2026-10-25-blockperf/smalln/` |
| `sgd_update` of `w1` at the bandwidth of `_foreach_add_` | the rest | same | not measured; at most −20.6 µs, its whole difference | — |
| the 13 floats per invocation site 12 spills to shared memory | big matmuls | same | not measured; site 12 is +345.2 against cuBLAS | `../2026-10-25-blockperf/diagnosis.md` |

## Back-to-back reference (`reference/`)

One session, 07:54–07:56 UTC, the sweep harness's own point runs, the GPU alone before each
(`reference/apps-before.txt`): the base bench, then `opponents/mlp_sweep/mlp_sweep.py --point
8192:4096 --modes eager`, then the bench with the loss-sum probe. Each run is
`--min-seconds 6`; the figure is the median of 3 timed runs. SM clock after each run's first
second, from `reference/clocks.csv`:

| side | runs, µs per step | median | SM MHz |
|---|---|---|---|
| modelspec base (2d83128) | 7,255.6 / 7,293.8 / 7,328.8 | 7,293.8 | 1,665–1,740 |
| PyTorch eager | 6,616.5 / 6,562.7 / 6,565.0 | 6,565.0 | 1,590–1,650 |
| modelspec with the loss-sum probe | 7,050.5 / 7,023.2 / 7,077.6 | 7,050.5 | 1,650–1,695 |

Base / PyTorch 1.111; probe / PyTorch 1.074. Even with the probe, modelspec is 485.5 µs per step
behind PyTorch here, and PyTorch ran at the lower clocks.
