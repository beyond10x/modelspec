---
format: aep.planning-md/3
id: verification-report:bert-tiny-vs-pytorch-2026-11-05-segment-reuse
kind: verification-report
status: draft
title: 'BERT-tiny pre-training on the RTX 3090 at 63000d0e: ModelSpec 3.70 ms/step against torch.compile 4.11 ms and PyTorch eager 7.52 ms, every ModelSpec run under every torch.compile run per step and to the baseline'
summary: 'Third series of 2026-11-05, at commit 63000d0e with recorded segments submitted again, 3 runs per side of 10,000 steps with the GPU in P0 or P2 at 1,700 MHz SM or more: ModelSpec 0.900x torch.compile and 0.492x eager per step, 0.893x and 0.488x to the held-out unigram baseline; ModelSpec''s slowest run is 8.0 % under torch.compile''s fastest per step and 8.4 % to the baseline; clocks not locked, 1-minute load 1.5 to 9.1'
relations:
- reviews: epic:bert-tiny-beats-torch-compile
revision: 2
---
# BERT-tiny pre-training on the RTX 3090 at 63000d0e: ModelSpec against torch.compile and PyTorch eager

Reviews `epic:bert-tiny-beats-torch-compile` (step F of `initiative:bert-tiny-beats-pytorch`): the
median µs per step of `examples/bert_tiny.mspec` at batch 32 on Vulkan and its training seconds
to the held-out unigram baseline against `torch.compile`'s, measured the same day on the same
device with the harness `opponents/bert_tiny_pretrain/`, three timed runs per side in turn. The
build is commit 63000d0e on the branch `perf/beats-pytorch`: release 0.25.0 (acf24d85) plus the
committed changes of `story:vulkan-cross-entropy-row-workgroup`, `story:vulkan-long-k-from-4096`,
`story:vulkan-embedding-sort-lanes`, `story:vulkan-adamw-one-dispatch-per-parameter`,
`story:vulkan-attention-blocked-key-tiles`, `story:vulkan-matmul-split-k` and
`story:vulkan-segment-command-buffer-reuse`. It is the third published series of 2026-11-05;
`verification-report:bert-tiny-vs-pytorch-2026-11-05` holds the first, at 30247795, and
`verification-report:bert-tiny-vs-pytorch-2026-11-05-split-k` the second, at 1cf9e12f.

**Verdict: every ModelSpec run of this series is under every `torch.compile` run, per step and to
the held-out baseline.** Medians of 3 runs of 10,000 steps: ModelSpec Vulkan 3,697.1 µs per step,
`torch.compile` 4,109.6, PyTorch CUDA eager 7,518.0. ModelSpec's three runs measured 3,641.5,
3,697.1 and 3,768.8 µs; `torch.compile`'s 4,096.0, 4,109.6 and 4,118.4. ModelSpec's slowest run is
327.2 µs (8.0 %) under `torch.compile`'s fastest. To the held-out baseline ModelSpec's runs took
3.962, 4.024 and 4.060 training seconds and `torch.compile`'s 4.432, 4.504 and 4.507; ModelSpec's
slowest is 0.372 s (8.4 %) under `torch.compile`'s fastest. All three sides first reach the
baseline at step 1,100. The epic's outcome, a median below `torch.compile`'s on both figures,
holds on these records, and in this series without an overlap between the runs of the two sides.

What the records do not show: a margin beyond this series. It is 3 runs per side with the
clocks not locked. The series ran from 15:00:08Z to 15:15:02Z in a window in which the other
sessions on the host held their builds (1-minute load 1.5 to 9.1); the series at 1cf9e12f ran at a
load of 6.3 to 33.5. ModelSpec's µs per step rose from run to run (3,641.5, 3,697.1, 3,768.8)
while the median SM clock of its runs fell (1890, 1845, 1800 MHz) and the GPU reached 86 to 88 °C;
`torch.compile`'s rose by less (4,096.0, 4,109.6, 4,118.4) over the same clock medians. That the
falling clock is the cause is inferred from these two series of figures, not measured.

## What changed since 1cf9e12f

One commit, 63000d0e (`story:vulkan-segment-command-buffer-reuse`): the stepwise runner records
the command buffer of a segment of training steps once per upload buffer and step count and
submits it again, where it made and destroyed a command pool and a command buffer per segment.
In the series at 1cf9e12f each ModelSpec run had 8 or 9 windows of 100 steps above 1.15 times
the run's median, 3.73 to 4.25 s of the run; in this series no run of any side has one
(ModelSpec's 100-step windows: 3,575 to 3,930, 3,642 to 3,781 and 3,646 to 3,900 µs per step).
The kernels are those of 1cf9e12f: ModelSpec's 101 held-out losses are the same printed digits
in all 3 runs and the same as in the series at 1cf9e12f.

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

The commands and script of `verification-report:bert-tiny-vs-pytorch-2026-11-05`: before the
series `xset dpms force on`, `xset -dpms` and `xset s off`; before each run `xset s reset`, 5 s,
and `xset q` must report `Monitor is On` or `DPMS is Disabled` (`display.txt`: `DPMS is
Disabled` before every run); after it `xset +dpms` and `xset s on`. Each run holds `flock` on
`~/.cache/modelspec-gpu.lock`. The runs alternate ModelSpec, eager, compile, three times.
`<packed-128>` is the packed WikiText-103 directory of `story:bert-tiny-gpu-pretraining` (binding
`668f3433-dbd7-8871-a5a7-a421c2bd1816`). A sampler appended the 1-minute load average to
`load.txt` every 10 s, and `uptime.txt` holds `uptime` before and after every run.

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 615.71.09, clocks not locked |
| ModelSpec | `modelspec 0.25.0` at 63000d0e838ae53e6af088b7a80083bcd4e8f119, `dirty: false`; release binary SHA-256 `b289ed00c845f3cc7c608eba2f986d1c969ca4beb2f211ffa5b06a6dfe7d421c` (`commit.txt`) |
| PyTorch | torch 2.15.0+cu132, CUDA 13.2, Python 3.14.7, tf32 off, torch.compile mode default (`pytorch-version.txt`) |
| other compute processes | none before each of the 11 GPU runs (`gpu-procs-<run>.txt`, header only) |
| clock samples | 1,799, every 500 ms, 16:00:03 to 16:15:02 local (UTC+1) on 2026-11-05 |
| load samples | 90, every 10 s |
| GPU temperature | 71 to 86 °C at the 10th second of a run, at most 85 to 88 °C in it (`clocks.csv`) |

## Comparability

| | ModelSpec Vulkan | PyTorch eager and compile | same |
|---|---|---|---|
| steps | 10,000 (closing line `steps 10000`) | 10,000 (result line `"steps": 10000`) | yes |
| held-out evaluation | step 0 and every 100 steps, 101 `eval` lines per run | 101 evaluations per run (102 JSON lines with the result line) | yes |
| batch and sequence | 32 sequences of 128, 4,096 tokens | the same (result line) | yes |
| data | binding `668f3433-…` | the same directory, ModelSpec's draw | yes |
| initial parameters | the 41 `init` lines of each of the 3 timed runs equal those of the `--steps 0` run (SHA-256 of the block `9c64ca8bcfd789819a2eb691a2d498511a094fc68d576a7c3fd6f7eeb7cd64cf`, as in the earlier series) | `initial.msckpt`, SHA-256 `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704`, byte-identical to the earlier series' | yes |
| step-0 held-out loss | 9.029766 in all 3 runs and the `--steps 0` run | 9.029766850451333 in all 6 runs; `--loss-check` 9.029767 | yes, within the printed digit |
| target | 6.926537926202018 | 6.926537926202018 | yes |
| FLOP per step | 37,446,746,112, `MatmulOnly` | 37,446,746,112, `MatmulOnly` | yes |
| training clock | per step, from recording or resubmitting the step's command buffer to the return of its host wait (`--clock`); batch draw and staging, plan build and pipeline creation excluded; the 101 held-out evaluations excluded, timed into `eval_seconds` (28.0, 28.2, 28.6 s) | per step, from a CUDA synchronize to one after the scheduler's step; evaluations and batch draw excluded | yes |
| GPU clocks, counted runs | SM 210–1935 MHz, median 1860 (n = 404); state P0; memory median 9751 MHz | eager SM 210–1935, median 1875 (n = 704); compile 210–1920, median 1845 (n = 513); state P2; memory median 9501 MHz | each side at its driver's full-power state |
| warm-up before the clock | none | 3 forward and backward passes and one AdamW step on a copy | differs, as before |
| dropout | ModelSpec's Philox stream | torch's own RNG | differs; the held-out curves agree within 2.3e-3 nats (below) |
| arithmetic | f32; 13 matmuls of the step split along `k` and summed in ascending slice order (DESIGN.md ADR-010), in the tolerance class of DESIGN.md § 5.4 | f32, tf32 off | both f32 |
| host load (1-minute) | 1.5 to 7.2, median 3.4 | eager 1.9 to 9.1, median 6.1; compile 4.8 to 8.8, median 7.3 | differs, per run below |

## The clock gate

After each run, the median SM clock and the most frequent performance state of the clock samples
inside its span (`run-clocks.txt`). A run with a median under 1,700 MHz or a state other than P0
and P2 would be re-run once; none was.

| run | median SM MHz | state | gate |
|---|---|---|---|
| modelspec-1 | 1890 | P0 | passed |
| pytorch-eager-1 | 1890 | P2 | passed |
| pytorch-compile-1 | 1890 | P2 | passed |
| modelspec-2 | 1845 | P0 | passed |
| pytorch-eager-2 | 1875 | P2 | passed |
| pytorch-compile-2 | 1845 | P2 | passed |
| modelspec-3 | 1800 | P0 | passed |
| pytorch-eager-3 | 1845 | P2 | passed |
| pytorch-compile-3 | 1800 | P2 | passed |

## Results (median of 3 runs of 10,000 steps, each figure its own median)

| side | µs per step | tokens/s | GFLOP/s | steps to target | training s to target | final held-out loss |
|---|---|---|---|---|---|---|
| ModelSpec Vulkan | 3,697.139 | 1,107,884 | 10,128.6 | 1,100 | 4.024 | 6.767498 |
| PyTorch CUDA eager | 7,517.975 | 544,828 | 4,981.0 | 1,100 | 8.252 | 6.767258 |
| PyTorch CUDA torch.compile | 4,109.567 | 996,699 | 9,112.1 | 1,100 | 4.504 | 6.767507 |

Ratios of medians: ModelSpec / compile 0.900 per step and 0.893 to the target; ModelSpec /
eager 0.492 per step and 0.488 to the target. ModelSpec's slowest run over `torch.compile`'s
fastest: 0.920 per step and 0.916 to the target. Against the series at 1cf9e12f (4,130.3 µs):
0.895. Against the first series of 2026-11-05 at 30247795 (5,619.4 µs): 0.658. Against 2026-10-29
(15,245.2 µs with the evaluations inside the clock): 0.243.

## Each run

| run | µs per step | training s | training s to target | final loss | load (1-min) min–max | 100-step windows, µs per step min–max |
|---|---|---|---|---|---|---|
| modelspec-1 | 3,641.533 | 36.415 | 3.962 | 6.767498 | 1.5–2.0 | 3,575–3,930 |
| modelspec-2 | 3,697.139 | 36.971 | 4.024 | 6.767498 | 2.9–7.2 | 3,642–3,781 |
| modelspec-3 | 3,768.793 | 37.688 | 4.060 | 6.767498 | 2.6–7.0 | 3,646–3,900 |
| pytorch-eager-1 | 7,593.043 | 75.930 | 8.441 | 6.7672577 | 1.9–9.1 | 7,411–7,870 |
| pytorch-eager-2 | 7,517.975 | 75.180 | 8.252 | 6.7672577 | 2.6–7.0 | 7,350–7,690 |
| pytorch-eager-3 | 7,461.353 | 74.614 | 8.178 | 6.7672577 | 2.6–8.4 | 7,244–8,066 |
| pytorch-compile-1 | 4,095.971 | 40.960 | 4.504 | 6.7675071 | 6.5–8.8 | 4,029–4,234 |
| pytorch-compile-2 | 4,109.567 | 41.096 | 4.507 | 6.7675071 | 4.8–8.1 | 4,032–4,325 |
| pytorch-compile-3 | 4,118.426 | 41.184 | 4.432 | 6.7675071 | 6.4–8.7 | 3,954–4,357 |

The record keeps every run as measured; no run was dropped or repeated. The load rose during
each PyTorch run and fell during each ModelSpec run (`load.txt`); `modelspec-1`, the only
ModelSpec run not preceded by a PyTorch run, saw 1.5 to 2.0. That the PyTorch process's own
threads are that load is inferred, not measured.

ModelSpec's training seconds to the target follow its µs per step in this series: 3,601, 3,658
and 3,691 µs per step up to step 1,100 against 3,642, 3,697 and 3,769 over the run. In the series
at 1cf9e12f they did not (4.058 to 4.663 s), and the trace of
`story:vulkan-segment-command-buffer-reuse` puts that spread in the waits for single segments
that this build no longer has.

## Held-out loss curves

ModelSpec's 101 held-out losses are identical across its 3 runs and identical to those of the
series at 1cf9e12f. The PyTorch runs agree with each other within 1.4e-8 per mode, and the first
run of each mode with that of the series at 1cf9e12f within 1.2e-8. Between sides: |ModelSpec − eager| at most 2.21e-3 nats,
|ModelSpec − compile| 1.09e-3, |eager − compile| 2.62e-3.

| step | ModelSpec | eager | compile |
|---|---|---|---|
| 0 | 9.029766 | 9.029767 | 9.029767 |
| 1,000 | 6.928237 | 6.928291 | 6.928193 |
| 1,100 | 6.923570 | 6.923565 | 6.923509 |
| 5,000 | 6.846160 | 6.845315 | 6.845205 |
| 10,000 | 6.767498 | 6.767258 | 6.767507 |

## Evidence

The raw records are in `benchmarks/raw/2026-11-05-bert-tiny-pretrain-4/`, laid out as those of
the earlier series of 2026-11-05: `modelspec-<n>.stdout.txt` (stdout without its `init` and
`param` lines), `modelspec-<n>.jsonl` (the `--record` event files), `pytorch-<mode>-<n>.jsonl`
and `pytorch-<mode>-<n>.stderr.txt` (the venv path replaced by `<venv>`), `loss-modelspec.txt`,
`loss-pytorch.txt`, `loss-pytorch.stderr.txt`, `clocks.csv` (local time, UTC+1), `load.txt`,
`times.txt`, `uptime.txt`, `display.txt`, `run-clocks.txt` (with the gate verdicts),
`commit.txt` (commit, version, dirty flag and binary SHA-256), `driver.txt`,
`pytorch-version.txt` and `gpu-procs-<run>.txt`. The opponent's result lines carry
`"dirty": false`, so the docs test `every_bert_tiny_pretrain_entry_repeats_the_raw_records_it_cites`
admits the three rows of `benchmarks/runs.yaml` that cite them.

| file | SHA-256 |
|---|---|
| `modelspec-1.jsonl` | `cff13d2d948cc97827764cf6a1baef1956822a9417081786b56de75081cbaf31` |
| `modelspec-2.jsonl` | `603381f18aef52bc186d6e78e1d1466f2550f368315e8d625603a78fb5c946ec` |
| `modelspec-3.jsonl` | `f509be329e4f48db2cc8710f39fa5fc28255dc4a19786234163178c9ab06a02e` |
| `pytorch-eager-1.jsonl` | `0683cdb0b85ddd92eaa8fa52a61530079afbfb4baa8fda9916ad03d78e7eec93` |
| `pytorch-eager-2.jsonl` | `27f7298ea0ac71f174f0a12e9609e78c90ae028662f9abfeb7c0af34ea71ed8a` |
| `pytorch-eager-3.jsonl` | `2c4144084ae493e6d45ae8e1f6a30738ab882722b87dbe27b0eb50d2b4ab8b62` |
| `pytorch-compile-1.jsonl` | `b8fb5f0bca621e6acae4d64fcdfa827b33791674289a301dcc5287a3ece1935e` |
| `pytorch-compile-2.jsonl` | `5628a085769702470c28209da9a827d630540c4ec949c8a886b0a1cf64e48a13` |
| `pytorch-compile-3.jsonl` | `290db48d59a0925248ef3739620f97c554a04aedee619c7d8aceb293dcff620a` |
| `initial.msckpt` (not committed, 5,951,104 bytes) | `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704` |
