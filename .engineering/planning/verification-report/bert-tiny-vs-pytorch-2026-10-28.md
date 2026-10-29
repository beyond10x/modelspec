---
format: aep.planning-md/3
id: verification-report:bert-tiny-vs-pytorch-2026-10-28
kind: verification-report
status: draft
title: 'BERT-tiny pre-training on the RTX 3090: ModelSpec 222.6 ms/step against PyTorch eager 9.9 ms and compile 4.9 ms, all three at the baseline by step 1,100'
summary: 'Unit 23 timed runs at 3b9869f, 3 runs per side of 10,000 steps: eager 22.5x and compile 45.5x faster per step, 25.0x and 50.4x faster to the held-out unigram baseline 6.926537926202018; host loaded (1-minute load 6.2 to 75.5); step time under investigation'
relations:
- reviews: story:bert-tiny-vs-pytorch-cuda
revision: 1
---
# BERT-tiny pre-training on the RTX 3090: ModelSpec against PyTorch eager and torch.compile

Reviews `story:bert-tiny-vs-pytorch-cuda` (unit 23 of wave bert-1, pass B, coordinator decision
D87). The coordinator ran the commands below on the committed tree at 3b9869f (`dirty: false`)
through `timed-runs.sh`, kept with the unit's scratch, from 10:40:29 to 14:40:02 UTC on
2026-10-28: 3 rounds of ModelSpec Vulkan, then PyTorch CUDA eager, then `torch.compile`, each run
with the GPU to itself (`nvidia-smi --query-compute-apps` listed no process before any of the 9
runs) and clocks never locked.

The result is unfavourable to ModelSpec. Its median step takes 22.5 times PyTorch eager's and
45.5 times `torch.compile`'s, and it reaches the target 25.0 and 50.4 times later in training
seconds. All three sides first reach the target at the same evaluated step, 1,100. The cause of
ModelSpec's step time is under investigation; this report names none.

## Commands

```console
cargo build --release -p modelspec-cli --features vulkan
modelspec train examples/bert_tiny.mspec --backend vulkan --data <packed-128> --steps 0 --eval-every 100 --checkpoint initial.msckpt
python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py --data <packed-128> --loss-check initial.msckpt
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 500 > clocks.csv
modelspec train examples/bert_tiny.mspec --backend vulkan --data <packed-128> --eval-every 100 --clock --record modelspec-<n>.jsonl
python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py --data <packed-128> --mode eager --checkpoint initial.msckpt --target 6.926537926202018 > pytorch-eager-<n>.jsonl
python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py --data <packed-128> --mode compile --checkpoint initial.msckpt --target 6.926537926202018 > pytorch-compile-<n>.jsonl
```

`<packed-128>` is the packed WikiText-103 directory of `story:bert-tiny-gpu-pretraining`
(binding `668f3433-dbd7-8871-a5a7-a421c2bd1816`, sequences of 128). A sampler appended the
1-minute load average to `load.txt` every 10 s over all 9 runs.

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04, clocks not locked |
| ModelSpec | 0.21.0 at 3b9869fe878c34debcb5ecdbb30ebde32da1bc96, `dirty: false` |
| PyTorch | torch 2.14.1+cu130, CUDA 13.0, Python 3.14.7, tf32 off, torch.compile mode default |
| other compute processes | none before each of the 9 runs |
| clock samples | 28,533, every 500 ms, 10:41:57 to 14:40:02 UTC |
| load samples | 1,428, every 10 s, same span |

## Comparability

| | ModelSpec Vulkan | PyTorch eager and compile | same |
|---|---|---|---|
| steps | 10,000 (closing line `steps 10000`) | 10,000 (result line `"steps": 10000`) | yes |
| held-out evaluation | step 0 and every 100 steps, 101 evaluations | step 0 and every 100 steps, 101 evaluations | yes |
| batch and sequence | 32 sequences of 128, 4,096 tokens (the example's batch, `data … seq_len 128`) | 32 sequences of 128, 4,096 tokens (result line) | yes |
| data | binding `668f3433-…`, its `HeldOut` split of 2,189 sequences | the same directory, 1,049,703 `Train` and 2,189 `HeldOut` sequences, ModelSpec's draw (checked against the scalar copy at start) | yes |
| initial parameters | from the example's seed; the 41 `init` lines of each timed run equal those of the `--steps 0` run that wrote `initial.msckpt` (SHA-256 of the block `9c64ca8bcfd789819a2eb691a2d498511a094fc68d576a7c3fd6f7eeb7cd64cf` in all four) | `initial.msckpt`, SHA-256 `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704` | yes |
| step-0 held-out loss | 9.029767 in all 3 runs | 9.029766850451333 in all 6 runs | yes |
| target | 6.926537926202018 | 6.926537926202018 | yes |
| FLOP per step | 37,446,746,112, `MatmulOnly` | 37,446,746,112, `MatmulOnly` | yes |
| training clock | per step, from recording the step's command buffer to the return of its host wait (`--clock`); held-out evaluations, batch draw and staging, plan build and pipeline creation excluded | per step, from a CUDA synchronize to one after the scheduler's step; held-out evaluations and batch draw and staging excluded | the same span per step |
| warm-up before the clock | none | 3 forward and backward passes and one AdamW step on a copy: eager 0.34 to 0.58 s, compile 2.4 to 5.7 s | differs |
| dropout | ModelSpec's Philox stream | torch's own RNG | differs; the held-out curves agree within 2.3e-3 nats (below) |
| host load (1-minute) | 6.2 to 75.5, median 18.6 | eager 14.7 to 32.7, median 19.0; compile 14.4 to 50.7, median 35.0 | differs, per run below |

## Results (median of 3 runs of 10,000 steps)

µs per step is a run's training seconds over its 10,000 steps; tokens per second 4,096 · 10⁶ / µs
per step; GFLOP/s 37,446,746,112 / µs per step / 1000. Steps to target is the first evaluated step
whose held-out loss is at most the target; seconds to target the training seconds at that
evaluation.

| side | µs/step | tokens/s | GFLOP/s | steps to target | training seconds to target | training seconds | final held-out loss |
|---|---|---|---|---|---|---|---|
| ModelSpec Vulkan | 222,576.0 | 18,402.7 | 168.2 | 1,100 | 245.969 | 2,225.760 | 6.7674985 |
| PyTorch eager | 9,897.5 | 413,843.0 | 3,783.5 | 1,100 | 9.820 | 98.975 | 6.7672577 |
| PyTorch compile | 4,887.5 | 838,057.8 | 7,661.8 | 1,100 | 4.881 | 48.875 | 6.7675071 |

Ratios of the medians (PyTorch faster by):

| | eager / ModelSpec | compile / ModelSpec |
|---|---|---|
| µs per step | 22.5× | 45.5× |
| training seconds to target | 25.0× | 50.4× |

## Each run

| run | UTC span | µs/step | seconds to target | training seconds | eval seconds | SM MHz (min–max, median, n) | 1-minute load (min–max, median) |
|---|---|---|---|---|---|---|---|
| modelspec-1 | 10:41:57–11:54:07 | 222,576.0 | 243.203 | 2,225.760 | 2,093.076 | 705–1965, 1890, 8,653 | 6.19–53.35, 14.57 |
| pytorch-eager-1 | 11:54:07–11:56:32 | 8,770.9 | 10.708 | 87.709 | | 240–1920, 1890, 292 | 16.17–22.92, 19.64 |
| pytorch-compile-1 | 11:56:32–11:58:10 | 4,634.8 | 4.881 | 46.348 | | 1695–1905, 1860, 198 | 14.36–18.67, 15.20 |
| modelspec-2 | 11:58:10–13:11:46 | 220,421.5 | 251.535 | 2,204.215 | 2,199.367 | 705–1950, 1890, 8,820 | 6.49–75.47, 18.30 |
| pytorch-eager-2 | 13:11:46–13:14:48 | 10,740.2 | 8.798 | 107.402 | | 285–1920, 1905, 366 | 14.71–28.62, 22.41 |
| pytorch-compile-2 | 13:14:48–13:16:33 | 4,887.5 | 4.778 | 48.875 | | 1725–1920, 1860, 211 | 21.69–33.29, 24.66 |
| modelspec-3 | 13:16:33–14:33:51 | 223,085.3 | 245.969 | 2,230.853 | 2,393.134 | 600–1935, 1890, 9,265 | 9.78–57.17, 23.53 |
| pytorch-eager-3 | 14:33:51–14:36:39 | 9,897.5 | 9.820 | 98.975 | | 300–1935, 1890, 337 | 16.01–32.72, 18.21 |
| pytorch-compile-3 | 14:36:39–14:40:02 | 7,932.0 | 16.555 | 79.320 | | 210–1935, 1890, 407 | 36.76–50.66, 40.12 |

The spans are whole processes (start-up, warm-up and held-out evaluations included), so the SM
clock figures cover those phases too, not the training steps alone. `pytorch-compile-3` ran at the highest load of the 9
runs and is the slowest compile run, 1.6 times the compile median; it is kept, and it is not the median. The three ModelSpec
runs print the same held-out losses at every evaluation; the PyTorch runs of one mode differ from each other
by at most 2.6e-9 of the loss. ModelSpec's curve is within 2.3e-3 nats of eager's (largest at step 6,900) and 1.1e-3 of compile's (step 7,200).

## Held-out loss curves

Run 1 of each side, every evaluation. ModelSpec as printed by `train`, PyTorch rounded to 6
decimals. The target is 6.926537926202018; all three are above it at step 1,000 and at or below
it from step 1,100.

| step | ModelSpec Vulkan | PyTorch eager | PyTorch compile |
|---|---|---|---|
| 0 | 9.029767 | 9.029767 | 9.029767 |
| 100 | 9.00717 | 9.007236 | 9.007154 |
| 200 | 8.881183 | 8.880979 | 8.880712 |
| 300 | 8.553788 | 8.554231 | 8.552970 |
| 400 | 8.160546 | 8.161860 | 8.160134 |
| 500 | 7.7516637 | 7.753552 | 7.751490 |
| 600 | 7.370238 | 7.372260 | 7.369639 |
| 700 | 7.0944147 | 7.095749 | 7.093660 |
| 800 | 6.9724994 | 6.972939 | 6.972154 |
| 900 | 6.938306 | 6.938399 | 6.938214 |
| 1000 | 6.928237 | 6.928291 | 6.928193 |
| 1100 | 6.9235697 | 6.923565 | 6.923509 |
| 1200 | 6.9203305 | 6.920331 | 6.920304 |
| 1300 | 6.919305 | 6.919314 | 6.919295 |
| 1400 | 6.919493 | 6.919506 | 6.919491 |
| 1500 | 6.9195824 | 6.919646 | 6.919563 |
| 1600 | 6.9175363 | 6.917525 | 6.917550 |
| 1700 | 6.9157243 | 6.915657 | 6.915693 |
| 1800 | 6.9150496 | 6.915008 | 6.915000 |
| 1900 | 6.9151974 | 6.915160 | 6.915214 |
| 2000 | 6.913918 | 6.913819 | 6.913776 |
| 2100 | 6.914126 | 6.913906 | 6.913890 |
| 2200 | 6.913253 | 6.912848 | 6.912873 |
| 2300 | 6.909996 | 6.909517 | 6.909396 |
| 2400 | 6.907103 | 6.906377 | 6.906401 |
| 2500 | 6.9030714 | 6.902241 | 6.902123 |
| 2600 | 6.899245 | 6.898470 | 6.898360 |
| 2700 | 6.896695 | 6.895983 | 6.895816 |
| 2800 | 6.8944488 | 6.893788 | 6.893668 |
| 2900 | 6.8924627 | 6.891703 | 6.891592 |
| 3000 | 6.8897038 | 6.889219 | 6.889122 |
| 3100 | 6.8877654 | 6.887356 | 6.887227 |
| 3200 | 6.886344 | 6.885649 | 6.885370 |
| 3300 | 6.884075 | 6.883535 | 6.883412 |
| 3400 | 6.8817005 | 6.881217 | 6.881064 |
| 3500 | 6.880594 | 6.880327 | 6.880098 |
| 3600 | 6.878766 | 6.878605 | 6.878383 |
| 3700 | 6.8784046 | 6.877796 | 6.877650 |
| 3800 | 6.875513 | 6.875175 | 6.874735 |
| 3900 | 6.8739924 | 6.873527 | 6.873405 |
| 4000 | 6.871486 | 6.871094 | 6.870855 |
| 4100 | 6.868565 | 6.868397 | 6.868031 |
| 4200 | 6.866189 | 6.865996 | 6.865457 |
| 4300 | 6.8645887 | 6.864576 | 6.863862 |
| 4400 | 6.861665 | 6.861185 | 6.860903 |
| 4500 | 6.859753 | 6.859029 | 6.858960 |
| 4600 | 6.856666 | 6.856087 | 6.855825 |
| 4700 | 6.8539453 | 6.853324 | 6.853204 |
| 4800 | 6.8516517 | 6.851078 | 6.850805 |
| 4900 | 6.848846 | 6.848115 | 6.848117 |
| 5000 | 6.84616 | 6.845315 | 6.845205 |
| 5100 | 6.842681 | 6.842062 | 6.841982 |
| 5200 | 6.840537 | 6.839773 | 6.839684 |
| 5300 | 6.8379703 | 6.837583 | 6.837267 |
| 5400 | 6.8351007 | 6.834738 | 6.834441 |
| 5500 | 6.832896 | 6.832285 | 6.832133 |
| 5600 | 6.8300724 | 6.829736 | 6.829549 |
| 5700 | 6.8277473 | 6.827381 | 6.827505 |
| 5800 | 6.824762 | 6.824019 | 6.824278 |
| 5900 | 6.822185 | 6.821352 | 6.821944 |
| 6000 | 6.818888 | 6.818300 | 6.818460 |
| 6100 | 6.8170724 | 6.815919 | 6.816591 |
| 6200 | 6.814433 | 6.813022 | 6.813810 |
| 6300 | 6.8122673 | 6.810867 | 6.811746 |
| 6400 | 6.8097634 | 6.808067 | 6.809395 |
| 6500 | 6.8074346 | 6.805917 | 6.806663 |
| 6600 | 6.805016 | 6.803340 | 6.804234 |
| 6700 | 6.8037543 | 6.801997 | 6.802825 |
| 6800 | 6.8013697 | 6.799551 | 6.800553 |
| 6900 | 6.8000436 | 6.797829 | 6.799174 |
| 7000 | 6.798351 | 6.796389 | 6.797368 |
| 7100 | 6.7963314 | 6.794231 | 6.795257 |
| 7200 | 6.794686 | 6.792526 | 6.793600 |
| 7300 | 6.792917 | 6.791016 | 6.792077 |
| 7400 | 6.7915025 | 6.789529 | 6.790749 |
| 7500 | 6.7896333 | 6.787826 | 6.788895 |
| 7600 | 6.7883058 | 6.786350 | 6.787615 |
| 7700 | 6.7867155 | 6.784736 | 6.786192 |
| 7800 | 6.7854137 | 6.783535 | 6.784845 |
| 7900 | 6.7840834 | 6.782115 | 6.783455 |
| 8000 | 6.782618 | 6.780796 | 6.782116 |
| 8100 | 6.7818294 | 6.780245 | 6.781438 |
| 8200 | 6.7801595 | 6.778677 | 6.779729 |
| 8300 | 6.7787046 | 6.777487 | 6.778495 |
| 8400 | 6.7779074 | 6.776630 | 6.777497 |
| 8500 | 6.776383 | 6.775194 | 6.776139 |
| 8600 | 6.7756057 | 6.774714 | 6.775445 |
| 8700 | 6.774767 | 6.773879 | 6.774584 |
| 8800 | 6.7736316 | 6.772993 | 6.773563 |
| 8900 | 6.772893 | 6.772246 | 6.772672 |
| 9000 | 6.771761 | 6.771295 | 6.771700 |
| 9100 | 6.7711244 | 6.770612 | 6.771032 |
| 9200 | 6.770505 | 6.770256 | 6.770535 |
| 9300 | 6.76978 | 6.769529 | 6.769637 |
| 9400 | 6.7693343 | 6.769077 | 6.769286 |
| 9500 | 6.7688212 | 6.768417 | 6.768645 |
| 9600 | 6.7682304 | 6.768026 | 6.768250 |
| 9700 | 6.7678533 | 6.767647 | 6.767865 |
| 9800 | 6.7676783 | 6.767444 | 6.767678 |
| 9900 | 6.7675323 | 6.767288 | 6.767535 |
| 10000 | 6.7674985 | 6.767258 | 6.767507 |

## Evidence

The raw records are in `benchmarks/raw/2026-10-28-bert-tiny-pretrain/`:
`modelspec-<n>.stdout.txt` (the `train` stdout of run `<n>` without its 41 `init` and 41 `param`
lines, the initial and final parameter values, about 39.6 MB per run), `modelspec-<n>.jsonl` (the
`--record` event files), `pytorch-<mode>-<n>.jsonl` and `pytorch-<mode>-<n>.stderr.txt` (the venv
path replaced by `<venv>`), `loss-modelspec.txt` (without its `init` and `param` lines),
`loss-pytorch.txt`, `loss-pytorch.stderr.txt`, `clocks.csv`, `load.txt`, `times.txt`,
`commit.txt`, `driver.txt`, `pytorch-version.txt` and `gpu-procs-<run>.txt`.

| file | SHA-256 |
|---|---|
| `modelspec-1.jsonl` | `8c57dd0ba9632cdfc14ed7d3bbcc1f4ef657354f78088168073bc43ffdcdb53f` |
| `modelspec-2.jsonl` | `35c4d63c7ae1e303e814d5b14188063c1592fe32ec8d0ec745e10a7c32316e26` |
| `modelspec-3.jsonl` | `454757db6a114be3dc396a2e810e3a7917dfcf1bfd51cefc190f02cee0a59385` |
| `pytorch-eager-1.jsonl` | `412482cc80d67be8ed4096e5e9915077c43e492542f64ef6d3c76732e5da5d4e` |
| `pytorch-eager-2.jsonl` | `8203314582b3fcdb9a3e43e40d9092cd0b273131c0ba61933e57856f888ca4cb` |
| `pytorch-eager-3.jsonl` | `24f300a26a198b3b3c52be4fd29d246d8ba25857456b187ed9a44ffd487a159a` |
| `pytorch-compile-1.jsonl` | `1cecae0b8c52bca39540fbd9dfa4616910286fe8659412588ca60aef6a1a4684` |
| `pytorch-compile-2.jsonl` | `798c0b36e8967af902648367e294e88fd76228481f4151fd6fd03f01f80a29d2` |
| `pytorch-compile-3.jsonl` | `a1a1f6f84955e70e8909ee8c00f78d90698a475dfbe367c733d283ce01c2544c` |
| `initial.msckpt` (not committed, 5,951,104 bytes) | `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704` |
| `modelspec-1.txt`, the unfiltered stdout (not committed) | `768a9adf54bfab91ab2d3a9a001d3d9cba2bd02f3c424b5339ea1169618095cf` |
| `modelspec-2.txt`, the unfiltered stdout (not committed) | `99b74a1bf4e2a6683488d1f8a458d1287ef3b71a346e8f0206b46276e372045b` |
| `modelspec-3.txt`, the unfiltered stdout (not committed) | `defeec2fad91b2e99132b007ea1e7e62f62d4c31991b4fe55d94abe3338f3895` |

`benchmarks/runs.yaml` holds one entry per side (`bert-tiny-pretrain-modelspec-vulkan-0.21.0`,
`bert-tiny-pretrain-pytorch-cuda-eager-0.21.0`, `bert-tiny-pretrain-pytorch-cuda-compile-0.21.0`)
with the three runs' µs per step in run order and the medians above; the `modelspec-docs` test
`every_bert_tiny_pretrain_entry_repeats_the_raw_records_it_cites` recomputes every one of them
from these files.
