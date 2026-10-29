---
format: aep.planning-md/3
id: verification-report:bert-tiny-vs-pytorch-2026-10-29
kind: verification-report
status: draft
title: 'BERT-tiny pre-training on the RTX 3090 after the step-time wave: ModelSpec 15.2 ms/step against PyTorch eager 7.8 ms and compile 4.2 ms, all three at the baseline by step 1,100'
summary: 'Timed runs at 3638e82, 3 counted runs per side of 10,000 steps with the GPU in P0 or P2 at 1,700 MHz SM or more, measured as on 2026-10-28: ModelSpec 14.6x faster per step and 14.8x faster to the held-out unigram baseline than at 3b9869f, still 1.95x eager and 3.61x compile per step and 1.91x and 3.50x to the baseline; its clock includes its held-out evaluations; an earlier series and two ModelSpec runs of this one ran in P3 with the display blanked and are not counted'
relations:
- reviews: epic:bert-tiny-step-tenth
revision: 1
---
# BERT-tiny pre-training on the RTX 3090 after the step-time wave: ModelSpec against PyTorch eager and torch.compile

Reviews `epic:bert-tiny-step-tenth` (the step-time wave: pipeline startup, evaluation metrics on
the device, the cross-entropy sum per row, the attention softmax statistics reused, the embedding
gradient per element, the MLM masked-row gather, the attention workgroup per head). The runs
repeat those of verification-report:bert-tiny-vs-pytorch-2026-10-28 on the integrated branch
`perf/bert-step-tenth` at 3638e82 (`dirty: false`; 3638e82 adds only planning-store files to the
code of 9e448ad), from a clean checkout of that commit, from 06:03:02 to 06:45:29 UTC on
2026-10-29: 3 rounds of ModelSpec Vulkan, then PyTorch CUDA eager, then `torch.compile`. Each run
took the GPU lock the other measuring sessions share for that run alone and found the GPU to
itself (`nvidia-smi --query-compute-apps` listed no process before any of the 13 GPU runs,
including the initial checkpoint, the loss check and the re-runs); clocks never locked. `uptime`
was recorded before and after each run.

Each run was gated on its GPU state (§ The clock gate): a run whose median SM clock was under
1,700 MHz or whose median performance state was neither P0 nor P2 was re-run, and the counted run
is the one that passed. Run 2 of ModelSpec was re-run twice; every other run passed at once. The
series this report first carried, from 04:03 to 04:56 UTC, ran ModelSpec in P3 throughout and is
not counted (§ The P3 runs).

The result is still unfavourable to ModelSpec. Its median step takes 0.0685 times its time of
2026-10-28 (14.6 times faster) and its training seconds to the target 0.0674 times (14.8 times
faster), but the step is still 1.95 times PyTorch eager's and 3.61 times `torch.compile`'s, and it
reaches the target 1.91 and 3.50 times later in training seconds. All three sides first reach the
target at step 1,100. One condition differs from 2026-10-28 and counts against ModelSpec
(§ Comparability): its training clock now includes its 101 held-out evaluations.

## Commands

```console
cargo build --release -p modelspec-cli --features vulkan
modelspec train examples/bert_tiny.mspec --backend vulkan --data <packed-128> --steps 0 --eval-every 100 --checkpoint initial.msckpt
python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py --data <packed-128> --loss-check initial.msckpt
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw --format=csv -lms 500 > clocks.csv
modelspec train examples/bert_tiny.mspec --backend vulkan --data <packed-128> --eval-every 100 --clock --record modelspec-<n>.jsonl
python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py --data <packed-128> --mode eager --checkpoint initial.msckpt --target 6.926537926202018 > pytorch-eager-<n>.jsonl
python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py --data <packed-128> --mode compile --checkpoint initial.msckpt --target 6.926537926202018 > pytorch-compile-<n>.jsonl
```

The commands are those of 2026-10-28, with the performance state added to the clock samples. Each
runs under `flock` on the shared GPU lock, and a run whose GPU is not free waits (no run had to).
Inside the lock and before the run, the X screen saver is reset (`xset s reset`, then 5 s), and
`xset q` must report `Monitor is On`, or `DPMS is Disabled`, which prints no monitor line; else
`xset dpms force on` and 5 s more (never needed). `display.txt` records what `xset q` reported and
the GPU's state at that moment. `<packed-128>` is the packed WikiText-103 directory of
`story:bert-tiny-gpu-pretraining` (binding `668f3433-dbd7-8871-a5a7-a421c2bd1816`, sequences of
128). A sampler appended the 1-minute load average to `load.txt` every 10 s over all runs.

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04, clocks not locked |
| ModelSpec | `modelspec 0.22.0` at 3638e82ed376c3fc6145f8f3f223e5834ac77a03, `dirty: false`, built in a clean checkout of that commit; the release binary's SHA-256 `59e8483c7b19d8f02007e143ddf291af193a45e1a3a4e2dc31f09a6f9bc80363` |
| PyTorch | torch 2.14.1+cu130, CUDA 13.0, Python 3.14.7, tf32 off, torch.compile mode default |
| other compute processes | none before each of the 13 GPU runs |
| clock samples | 5,786, every 500 ms, 06:01:43 to 06:50:40 UTC |
| load samples | 291, every 10 s, 06:01 to 06:50 UTC |

## Comparability

| | ModelSpec Vulkan | PyTorch eager and compile | same |
|---|---|---|---|
| steps | 10,000 (closing line `steps 10000`) | 10,000 (result line `"steps": 10000`) | yes, and as on 2026-10-28 |
| held-out evaluation | step 0 and every 100 steps, 101 evaluations | step 0 and every 100 steps, 101 evaluations | yes, and as on 2026-10-28 |
| batch and sequence | 32 sequences of 128, 4,096 tokens | 32 sequences of 128, 4,096 tokens (result line) | yes, and as on 2026-10-28 |
| data | binding `668f3433-…`, its `HeldOut` split of 2,189 sequences | the same directory, ModelSpec's draw | yes, and as on 2026-10-28 |
| initial parameters | the 41 `init` lines of each of the 5 timed runs equal those of the `--steps 0` run (SHA-256 of the block `9c64ca8bcfd789819a2eb691a2d498511a094fc68d576a7c3fd6f7eeb7cd64cf` in all six, as on 2026-10-28) | `initial.msckpt`, SHA-256 `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704`, byte-identical to 2026-10-28's | yes |
| step-0 held-out loss | 9.029766 in all 5 runs and the `--steps 0` run | 9.029766850451333 in all 6 runs; `--loss-check` 9.029767 | yes, within the printed digit |
| target | 6.926537926202018 | 6.926537926202018 | yes |
| FLOP per step | 37,446,746,112, `MatmulOnly` | 37,446,746,112, `MatmulOnly` | yes |
| training clock | per step, from recording the step's command buffer to the return of its host wait (`--clock`); the batch draw and staging, plan build and pipeline creation excluded; **the 101 held-out evaluations included** (below) | per step, from a CUDA synchronize to one after the scheduler's step; held-out evaluations and batch draw and staging excluded | differs, against ModelSpec |
| GPU clocks, counted runs | SM 210–1980 MHz, median 1905 (n = 942); median state P0; memory median 9751 MHz | eager SM 210–1920, median 1860 (n = 744), compile 210–1920, median 1860 (n = 564); median state P2; memory median 9501 MHz | each side at its driver's full-power state; on 2026-10-28 ModelSpec's median SM was 1890 MHz |
| warm-up before the clock | none | 3 forward and backward passes and one AdamW step on a copy | differs, as on 2026-10-28 |
| dropout | ModelSpec's Philox stream | torch's own RNG | differs; the held-out curves agree within 2.3e-3 nats (below) |
| host load (1-minute), counted runs | 3.7 to 18.7, median 6.3 | eager 2.6 to 15.5, median 9.5; compile 10.0 to 14.3, median 12.3 | differs, per run below |

The evaluations. Since story:vulkan-eval-metrics-on-device a Vulkan run reduces each held-out
evaluation on the device, and its batches open the evaluated step's command buffer
(`crates/modelspec-runtime/src/vulkan/stepwise.rs`: "Reduced on the device, the batches open the
step's command buffer"), so the training clock spans them and `eval_seconds` stays 0: every run
here closes with `eval_seconds 0.000000`, where the 2026-10-28 runs closed with 2,093 to 2,393 s.
How much of the clock they are was probed after the P3 series, with that series' binary (the
same code) under the same lock, 2 pairs of 2,000-step runs: with `--eval-every 100`
(21 evaluations) 64.042 and 62.517 training seconds, with `--eval-every 1000` (3 evaluations)
58.883 and 59.150. That is 0.237 s per evaluation (0.187 and 0.286 per pair). The probe ran in P3
like that series, so its figure does not carry over to the counted runs; no probe was run in P0.
The record keeps the figures as measured.

## The clock gate

After each run, the median SM clock and the median performance state of the clock samples inside
its span (`run-clocks.txt`). A run with a median SM clock under 1,700 MHz or a median state other
than P0 and P2 is off the gate and is re-run once; both stay in the raw records and the run that
passed is counted.

| run | median SM MHz | median state | gate | counted |
|---|---|---|---|---|
| modelspec-1 | 1920 | P0 | passed | yes |
| pytorch-eager-1 | 1860 | P2 | passed | yes |
| pytorch-compile-1 | 1860 | P2 | passed | yes |
| modelspec-2 | 900 | P3 | off | no |
| modelspec-2-r | 915 | P3 | off | no |
| pytorch-eager-2 | 1860 | P2 | passed | yes |
| pytorch-compile-2 | 1890 | P2 | passed | yes |
| modelspec-3 | 1905 | P0 | passed | yes |
| pytorch-eager-3 | 1890 | P2 | passed | yes |
| pytorch-compile-3 | 1860 | P2 | passed | yes |
| modelspec-2-r2 | 1890 | P0 | passed | yes, as run 2 |

`modelspec-2-r` was the one re-run the gate allows, and it was off the gate too. `modelspec-2-r2`
is a third attempt at run 2, made after the screen-saver reset was added to the lock (below); it
is outside the gate's once-only rule and is named here for that reason. The series was run in
three pieces (`continued` lines in `times.txt`): the first script's gate read the sample count
where the state belonged and called `modelspec-1` (1920 MHz, P0) off; it was stopped before the
needless re-run started, and the series continued with the gate corrected and `modelspec-1`
counted as measured.

The display. The coordinator found that the NVIDIA driver keeps a Vulkan client in P3 while the
monitor is asleep, and disabled DPMS before this series. `xset q` reported `DPMS is Disabled`
before each run up to `pytorch-eager-2` (06:31:58), and `Monitor is On` from `pytorch-compile-2`
(06:34:13) on, so DPMS was enabled again in between; who enabled it is not recorded. With DPMS
disabled, `modelspec-2` (06:19:46) and `modelspec-2-r` (06:26:17) still ran in P3 at 900 and
915 MHz. The X screen saver was at that time set to blank the screen after 600 s idle; from 06:34 every run first reset
it (`xset s reset`), and every ModelSpec run after that, `modelspec-3` and `modelspec-2-r2`, ran in
P0. That the screen saver's blanking put `modelspec-2` and `modelspec-2-r` in P3 is inferred from
this sequence, not observed: the screen's state during those runs was not recorded. `modelspec-1`
(06:05:47) ran in P0 without the reset.

## Results (median of the 3 counted runs of 10,000 steps)

µs per step is a run's training seconds over its 10,000 steps; tokens per second 4,096 · 10⁶ / µs
per step; GFLOP/s 37,446,746,112 / µs per step / 1000. Steps to target is the first evaluated step
whose held-out loss is at most the target; seconds to target the training seconds at that
evaluation.

| side | µs/step | tokens/s | GFLOP/s | steps to target | training seconds to target | training seconds | final held-out loss |
|---|---|---|---|---|---|---|---|
| ModelSpec Vulkan | 15,245.2 | 268,674.9 | 2,456.3 | 1,100 | 16.569 | 152.452 | 6.767498 |
| PyTorch eager | 7,806.5 | 524,687.6 | 4,796.8 | 1,100 | 8.654 | 78.065 | 6.7672577 |
| PyTorch compile | 4,220.6 | 970,482.2 | 8,872.4 | 1,100 | 4.734 | 42.206 | 6.7675071 |

Ratios of the medians:

| | ModelSpec / eager | ModelSpec / compile |
|---|---|---|
| µs per step | 1.95× | 3.61× |
| training seconds to target | 1.91× | 3.50× |

Against verification-report:bert-tiny-vs-pytorch-2026-10-28 (3b9869f, `modelspec 0.21.0`), the
medians of each side:

| side | µs per step, 2026-10-28 → 2026-10-29 | ratio | seconds to target | ratio |
|---|---|---|---|---|
| ModelSpec Vulkan | 222,576.0 → 15,245.2 | 0.0685 (14.6× faster) | 245.969 → 16.569 | 0.0674 (14.8× faster) |
| PyTorch eager | 9,897.5 → 7,806.5 | 0.789 | 9.820 → 8.654 | 0.881 |
| PyTorch compile | 4,887.5 → 4,220.6 | 0.864 | 4.881 → 4.734 | 0.970 |

PyTorch's own figures moved by up to 1.27× per step between the two days on the same opponent,
torch and driver: the host's load and the opponent's run-to-run spread, not a change of PyTorch.
ModelSpec's ratio to eager went from 22.5× to 1.95× per step and from 25.0× to 1.91× to the
target; to compile from 45.5× to 3.61× and from 50.4× to 3.50×.

Wall time, whole process (start-up, pipeline creation and evaluations included): a counted
ModelSpec run took 160, 156 and 155 s, a median of 15.6 ms per step, against 4,330, 4,416 and
4,638 s on 2026-10-28 (0.035 of the median). The epic's target, at most 43 ms of wall time per
step with evaluation, is met by all 3 counted runs, and by the two P3 runs of this series too
(299 and 295 s).

## Each run

| run | UTC span | µs/step | seconds to target | training seconds | SM MHz (min–max, median, n) | state (min–max, median) | memory MHz median | 1-minute load (min–max, median) | `uptime` load before → after |
|---|---|---|---|---|---|---|---|---|---|
| modelspec-1 | 06:05:47–06:08:27 | 15,490.9 | 17.838 | 154.909 | 450–1980, 1920, 320 | P0–P0, P0 | 9751 | 11.11–18.73, 15.67 | 12.35 → 10.46 |
| pytorch-eager-1 | 06:10:04–06:12:11 | 7,866.9 | 9.172 | 78.669 | 210–1860, 1860, 254 | P2–P5, P2 | 9501 | 7.08–15.49, 13.81 | 7.08 → 13.65 |
| pytorch-compile-1 | 06:12:20–06:13:51 | 4,260.4 | 4.734 | 42.604 | 210–1860, 1860, 182 | P2–P8, P2 | 9501 | 11.95–13.04, 12.70 | 14.00 → 12.89 |
| modelspec-2 (off, not counted) | 06:19:46–06:24:45 | 29,567.1 | 35.156 | 295.671 | 780–1035, 900, 598 | P3–P3, P3 | 5001 | 2.52–7.18, 5.05 | 7.18 → 3.39 |
| modelspec-2-r (off, not counted) | 06:26:17–06:31:12 | 29,200.0 | 34.904 | 292.000 | 780–1035, 915, 590 | P3–P3, P3 | 5001 | 2.32–8.52, 3.81 | 8.83 → 2.27 |
| pytorch-eager-2 | 06:31:58–06:33:59 | 7,675.6 | 8.369 | 76.756 | 210–1860, 1860, 242 | P2–P8, P2 | 9501 | 2.63–10.29, 7.40 | 2.69 → 11.47 |
| pytorch-compile-2 | 06:34:13–06:35:52 | 4,220.6 | 4.939 | 42.206 | 210–1920, 1890, 198 | P2–P8, P2 | 9501 | 10.00–12.44, 10.90 | 10.62 → 12.42 |
| modelspec-3 | 06:36:01–06:38:36 | 15,168.9 | 16.569 | 151.689 | 210–1935, 1905, 310 | P0–P8, P0 | 9751 | 4.16–10.55, 5.33 | 11.12 → 4.86 |
| pytorch-eager-3 | 06:38:45–06:40:49 | 7,806.5 | 8.654 | 78.065 | 210–1920, 1890, 248 | P2–P8, P2 | 9501 | 4.73–15.33, 9.62 | 4.86 → 15.26 |
| pytorch-compile-3 | 06:40:58–06:42:30 | 4,067.9 | 4.475 | 40.679 | 210–1920, 1860, 184 | P2–P8, P2 | 9501 | 11.68–14.30, 12.66 | 13.84 → 11.71 |
| modelspec-2-r2 (counted as run 2) | 06:42:53–06:45:29 | 15,245.2 | 16.536 | 152.452 | 210–1935, 1890, 312 | P0–P8, P0 | 9751 | 3.69–8.10, 4.88 | 8.63 → 5.81 |

The spans are whole processes, so the clock figures cover start-up, warm-up and evaluations too;
the P5 and P8 samples are the idle edges of a span. The machine ran other sessions' work
throughout; the gaps between runs are the waits for the shared GPU lock and the series' restart.
In run order, ModelSpec's counted run 2 is `modelspec-2-r2`, measured last.

## The P3 runs

The series this report first carried ran from 04:03:40 to 04:56:02 UTC on 2026-10-29 with the
same code (binary SHA-256 `4d159d16ce35d86d8cfc3b19a4332acb920121dafcda52168ff1ca892172061a`,
built in another checkout of 3638e82) and the same commands, without the screen-saver reset and
without the performance state in the clock samples. Its three ModelSpec runs ran at SM 210–885 MHz,
median 780 (n = 2,094), memory median 5001 MHz, the clocks of P3 (the probe after the series
recorded the state: P3 at 780 to 915 MHz SM and 5001 MHz memory), and took 34,825.6, 34,131.7 and
33,443.8 µs per step, median 34,131.7, 38.262 s to the target. Its PyTorch runs ran at a median
1860 MHz SM and 9501 MHz memory (eager median 11,449.3 µs per step, compile 6,964.2) under 1-minute
loads of 11.7 to 150.8. Not used, because the GPU ran ModelSpec at under half its SM clock and at
the lower memory clock, with the display blanked (the coordinator's finding: screen on, P0 at
1950 to 1965 MHz SM and the 200-step file in 2.98 s; screen off, P3 and 6.33 to 6.58 s), so the
series measured the display's power state rather than the wave. The whole series was replaced,
PyTorch's runs included, so that all three sides are measured in one window.
`modelspec-2` and `modelspec-2-r` of this series are P3 runs for the same reason (§ The clock
gate). The training bits of all P3 runs equal those of the counted runs (below).

## Held-out loss curves

All five ModelSpec runs of this series, counted or not, print the same held-out losses at every
evaluation, the same `step` losses and the same final `param` lines; their held-out losses equal
the P3 series', and their `step` and `param` lines are byte-identical to the 2026-10-28 runs':
the wave changed no training bit. The held-out losses differ from 2026-10-28's by at most 1e-6 in
the printed digits (step 0: 9.029766 against 9.029767), the device reduction's `Exp` and `Log` of
story:vulkan-eval-metrics-on-device (coordinator decision D128). The PyTorch runs of one mode
differ from each other by at most 1.1e-8. ModelSpec's curve is within 2.3e-3 nats of eager's
(largest at step 6,900) and 1.1e-3 of compile's (step 7,200), as on 2026-10-28. Run 1 of each
side, every 1,000 steps and the crossing at 1,100; ModelSpec as printed by `train`, PyTorch
rounded to 6 decimals:

| step | ModelSpec Vulkan | PyTorch eager | PyTorch compile |
|---|---|---|---|
| 0 | 9.029766 | 9.029767 | 9.029767 |
| 1000 | 6.928237 | 6.928291 | 6.928193 |
| 1100 | 6.9235697 | 6.923565 | 6.923509 |
| 2000 | 6.913918 | 6.913819 | 6.913776 |
| 3000 | 6.8897038 | 6.889219 | 6.889122 |
| 4000 | 6.871486 | 6.871094 | 6.870855 |
| 5000 | 6.84616 | 6.845315 | 6.845205 |
| 6000 | 6.818888 | 6.818300 | 6.818460 |
| 7000 | 6.798351 | 6.796389 | 6.797368 |
| 8000 | 6.7826176 | 6.780796 | 6.782116 |
| 9000 | 6.771761 | 6.771295 | 6.771700 |
| 10000 | 6.767498 | 6.767258 | 6.767507 |

## Evidence

The raw records are in `benchmarks/raw/2026-10-29-bert-tiny-pretrain/`, laid out as those of
2026-10-28, every run of the series including the two off the gate: `modelspec-<run>.stdout.txt`
(the `train` stdout without its 41 `init` and 41 `param` lines), `modelspec-<run>.jsonl` (the
`--record` event files), `pytorch-<mode>-<n>.jsonl` and `pytorch-<mode>-<n>.stderr.txt` (the venv
path replaced by `<venv>`), `loss-modelspec.txt` (without its `init` and `param` lines),
`loss-pytorch.txt`, `loss-pytorch.stderr.txt`, `clocks.csv` (local time, UTC+1), `load.txt`,
`times.txt`, `uptime.txt` (the `uptime` line before and after each run), `display.txt`,
`run-clocks.txt`, `commit.txt`, `driver.txt`, `pytorch-version.txt` and `gpu-procs-<run>.txt`. The
P3 series' records are in `benchmarks/raw/2026-10-29-bert-tiny-pretrain-p3/`, the same layout
without its three ModelSpec event files.

| file | SHA-256 |
|---|---|
| `modelspec-1.jsonl` | `87bedef77c44219b046151fcd85999a1035fafd11794561693c4465f3f5feaa5` |
| `modelspec-2.jsonl` (off the gate) | `78595e963aaf84b22dfd37d9d17784d7423961cc839e7ed73253d19db784f2ca` |
| `modelspec-2-r.jsonl` (off the gate) | `09c3f7466e875056fcd464c9b0ede1b7be9136bbd596d07f0901ce418892d05a` |
| `modelspec-2-r2.jsonl` | `890b36a6c56a10fe15d62dc23cf67c0f7025afab5614613186941b44ccd0b136` |
| `modelspec-3.jsonl` | `e84fe0b9754b9ef9809888fbadc43146a3881bf20d4393745fd6a94196d15e8f` |
| `pytorch-eager-1.jsonl` | `ecc85b2114fafbb3267bea80fc4a4ad80d1a082f41146bed3afe31574b3db0a7` |
| `pytorch-eager-2.jsonl` | `3548e9a5df979ea85bbe90af65f8cc2f0a28082fff1080630a07be85084a3883` |
| `pytorch-eager-3.jsonl` | `764c7bad911e5f0bd9f649635b814fa498eb8fc6b2603f4b991bb18ad2e08dc7` |
| `pytorch-compile-1.jsonl` | `ab6ffaaba944009db2c855431bda4a5045ace038c478cc5b6ade644e9885820f` |
| `pytorch-compile-2.jsonl` | `44752d8f3d3d51c6a69ee9554f2c47316e1b3d1fc649dd1dea45f5867f510d3a` |
| `pytorch-compile-3.jsonl` | `690776b35d0ab01caa80b685856bafb2b9a1663ef49f4083b0f15795fed41714` |
| `initial.msckpt` (not committed, 5,951,104 bytes) | `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704` |
| `modelspec-1.txt`, the unfiltered stdout (not committed) | `b6374001ab8df3f1be8e2c5d00947afc155311cb6ffb062e3b550ea8994a077a` |
| `modelspec-2.txt`, the unfiltered stdout (not committed) | `36396ed30703d9c36d00dd7477fd01ed11bf2abe5d2e90eb21c42f347365f4c8` |
| `modelspec-2-r.txt`, the unfiltered stdout (not committed) | `569b57a037672fea2187396bda6fd2bb928ea0d8f7037b1a7a744a69487ddfa8` |
| `modelspec-2-r2.txt`, the unfiltered stdout (not committed) | `7fe4f56c18fc70a8cf4a6a9d6570d243d8f74c18c218b0303ab2978ca0ffe346` |
| `modelspec-3.txt`, the unfiltered stdout (not committed) | `32c90e3178e02d59b57c212417eeaeaf2f6ae768cfd181481f8753c595d2f1da` |

`benchmarks/runs.yaml` holds one entry per side (`bert-tiny-pretrain-modelspec-vulkan-0.22.0`,
`bert-tiny-pretrain-pytorch-cuda-eager-0.22.0`, `bert-tiny-pretrain-pytorch-cuda-compile-0.22.0`)
beside the three of 2026-10-28, with the three counted runs' µs per step in run order and the
medians above; the ModelSpec entry cites `modelspec-1`, `modelspec-2-r2` and `modelspec-3`. The P3
runs are not in it. The ModelSpec entry's version is the binary's (`modelspec 0.22.0`, the version
of the tree before the release that will carry the wave), as the 2026-10-28 entries carry
`0.21.0`. The `modelspec-docs` test `every_bert_tiny_pretrain_entry_repeats_the_raw_records_it_cites`
recomputes every figure of all six entries from these files.
