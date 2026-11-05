---
format: aep.planning-md/3
id: verification-report:bert-tiny-vs-pytorch-2026-11-05-split-k
kind: verification-report
status: draft
title: 'BERT-tiny pre-training on the RTX 3090 at 1cf9e12f: ModelSpec 4.13 ms/step against torch.compile 4.31 ms and PyTorch eager 7.88 ms, every ModelSpec run under every torch.compile run per step'
summary: 'Second series of 2026-11-05, at commit 1cf9e12f with split-K, 3 runs per side of 10,000 steps with the GPU in P0 or P2 at 1,700 MHz SM or more: ModelSpec 0.958x torch.compile and 0.524x eager per step, 0.834x and 0.459x to the held-out unigram baseline; ModelSpec''s slowest run is 2.2 % under torch.compile''s fastest per step, and the seconds to the baseline overlap between runs; clocks not locked, 1-minute load 6.3 to 33.5'
relations:
- reviews: epic:bert-tiny-beats-torch-compile
revision: 2
---
# BERT-tiny pre-training on the RTX 3090 at 1cf9e12f: ModelSpec against torch.compile and PyTorch eager

Reviews `epic:bert-tiny-beats-torch-compile` (step F of `initiative:bert-tiny-beats-pytorch`): the
median µs per step of `examples/bert_tiny.mspec` at batch 32 on Vulkan and its training seconds
to the held-out unigram baseline against `torch.compile`'s, measured the same day on the same
device with the harness `opponents/bert_tiny_pretrain/`, three timed runs per side in turn. The
build is commit 1cf9e12f on the branch `perf/beats-pytorch`: release 0.25.0 (acf24d85) plus the
committed changes of `story:vulkan-cross-entropy-row-workgroup`, `story:vulkan-long-k-from-4096`,
`story:vulkan-embedding-sort-lanes`, `story:vulkan-adamw-one-dispatch-per-parameter`,
`story:vulkan-attention-blocked-key-tiles` and `story:vulkan-matmul-split-k`. It is the second
published series of 2026-11-05; `verification-report:bert-tiny-vs-pytorch-2026-11-05` holds the
first, at 30247795.

**Verdict: per step, every ModelSpec run of this series is under every `torch.compile` run.**
Medians of 3 runs of 10,000 steps: ModelSpec Vulkan 4,130.3 µs per step, `torch.compile` 4,311.7,
PyTorch CUDA eager 7,883.8. ModelSpec's three runs measured 4,118.9, 4,130.3 and 4,163.9 µs;
`torch.compile`'s 4,259.1, 4,311.7 and 5,117.7. ModelSpec's slowest run is 95.2 µs (2.2 %) under
`torch.compile`'s fastest. To the held-out baseline ModelSpec's median is 4.123 training seconds
against `torch.compile`'s 4.945; there the runs overlap (ModelSpec 4.058, 4.123 and 4.663 s,
`torch.compile` 4.614, 4.945 and 5.915 s). All three sides first reach the baseline at step
1,100. The epic's outcome, a median below `torch.compile`'s on both figures, holds on these
records.

What the records do not show: a margin beyond this series. It is 3 runs per side with the
clocks not locked on a host other sessions were loading (1-minute load 6.3 to 33.5), and
`torch.compile`'s run 1 ran at the highest load of its side. A series earlier the same day at
5ca80f2e, before split-K, measured ModelSpec at 4,589.6, 4,685.1 and 4,842.2 µs and
`torch.compile` at 4,576.2, 5,118.7 and 6,211.6; its ranges overlap, and it has no rows in
`benchmarks/runs.yaml` (raw records `benchmarks/raw/2026-11-05-bert-tiny-pretrain-2/`).

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
| ModelSpec | `modelspec 0.25.0` at 1cf9e12f583456617fa6ca1011a93bb5b861468f, `dirty: false`; release binary SHA-256 `0b6f489febd8ae2a869b3cb8a8d9162df7f9f713b5d5124c46c32a633783a3fa` (`commit.txt`) |
| PyTorch | torch 2.15.0+cu132, CUDA 13.2, Python 3.14.7, tf32 off, torch.compile mode default |
| other compute processes | none before each of the 11 GPU runs (`gpu-procs-<run>.txt`, header only) |
| clock samples | 2,067, every 500 ms, 11:40:04 to 11:57:18 local (UTC+1) on 2026-11-05 |
| load samples | 104, every 10 s |

## Comparability

| | ModelSpec Vulkan | PyTorch eager and compile | same |
|---|---|---|---|
| steps | 10,000 (closing line `steps 10000`) | 10,000 (result line `"steps": 10000`) | yes |
| held-out evaluation | step 0 and every 100 steps, 101 `eval` lines per run | 101 evaluations per run (102 JSON lines with the result line) | yes |
| batch and sequence | 32 sequences of 128, 4,096 tokens | the same (result line) | yes |
| data | binding `668f3433-…` | the same directory, ModelSpec's draw | yes |
| initial parameters | the 41 `init` lines of each of the 3 timed runs equal those of the `--steps 0` run (SHA-256 of the block `9c64ca8bcfd789819a2eb691a2d498511a094fc68d576a7c3fd6f7eeb7cd64cf`, as on 2026-10-29 and in the first series of 2026-11-05) | `initial.msckpt`, SHA-256 `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704`, byte-identical to 2026-10-29's | yes |
| step-0 held-out loss | 9.029766 in all 3 runs and the `--steps 0` run | 9.029766850451333 in all 6 runs; `--loss-check` 9.029767 | yes, within the printed digit |
| target | 6.926537926202018 | 6.926537926202018 | yes |
| FLOP per step | 37,446,746,112, `MatmulOnly` | 37,446,746,112, `MatmulOnly` | yes |
| training clock | per step, from recording the step's command buffer to the return of its host wait (`--clock`); batch draw and staging, plan build and pipeline creation excluded; the 101 held-out evaluations excluded, timed into `eval_seconds` (20.4, 26.1, 26.6 s) | per step, from a CUDA synchronize to one after the scheduler's step; evaluations and batch draw excluded | yes |
| GPU clocks, counted runs | SM 210–1935 MHz, median 1875 (n = 405); state P0; memory median 9751 MHz | eager SM 210–1935, median 1890 (n = 884); compile 210–1920, median 1860 (n = 592); state P2; memory median 9501 MHz | each side at its driver's full-power state |
| warm-up before the clock | none | 3 forward and backward passes and one AdamW step on a copy | differs, as before |
| dropout | ModelSpec's Philox stream | torch's own RNG | differs; the held-out curves agree within 2.3e-3 nats (below) |
| arithmetic | f32; 13 matmuls of the step split along `k` and summed in ascending slice order (DESIGN.md ADR-010), in the tolerance class of DESIGN.md § 5.4 | f32, tf32 off | both f32 |
| host load (1-minute) | 7.1 to 22.3, median 10.9 | eager 6.3 to 33.5, median 11.5; compile 10.5 to 30.8, median 11.7 | differs, per run below |

## The clock gate

After each run, the median SM clock and the most frequent performance state of the clock samples
inside its span (`run-clocks.txt`). A run with a median under 1,700 MHz or a state other than P0
and P2 would be re-run once; none was.

| run | median SM MHz | state | gate |
|---|---|---|---|
| modelspec-1 | 1920 | P0 | passed |
| pytorch-eager-1 | 1905 | P2 | passed |
| pytorch-compile-1 | 1890 | P2 | passed |
| modelspec-2 | 1845 | P0 | passed |
| pytorch-eager-2 | 1875 | P2 | passed |
| pytorch-compile-2 | 1845 | P2 | passed |
| modelspec-3 | 1800 | P0 | passed |
| pytorch-eager-3 | 1860 | P2 | passed |
| pytorch-compile-3 | 1830 | P2 | passed |

## Results (median of 3 runs of 10,000 steps, each figure its own median)

| side | µs per step | tokens/s | GFLOP/s | steps to target | training s to target | final held-out loss |
|---|---|---|---|---|---|---|
| ModelSpec Vulkan | 4,130.349 | 991,684 | 9,066.2 | 1,100 | 4.123 | 6.767498 |
| PyTorch CUDA eager | 7,883.785 | 519,547 | 4,749.8 | 1,100 | 8.976 | 6.767258 |
| PyTorch CUDA torch.compile | 4,311.663 | 949,981 | 8,685.0 | 1,100 | 4.945 | 6.767507 |

Ratios of medians: ModelSpec / compile 0.958 per step and 0.834 to the target; ModelSpec /
eager 0.524 per step and 0.459 to the target. ModelSpec's slowest run over `torch.compile`'s
fastest: 0.978. Against the first series of 2026-11-05 at 30247795 (5,619.4 µs): 0.735. Against
2026-10-29 (15,245.2 µs with the evaluations inside the clock): 0.271.

## Each run

| run | µs per step | training s | training s to target | final loss | load (1-min) min–max |
|---|---|---|---|---|---|
| modelspec-1 | 4,118.855 | 41.189 | 4.058 | 6.767498 | 10.6–11.3 |
| modelspec-2 | 4,130.349 | 41.303 | 4.123 | 6.767498 | 10.3–22.3 |
| modelspec-3 | 4,163.905 | 41.639 | 4.663 | 6.767498 | 7.1–11.4 |
| pytorch-eager-1 | 10,978.603 | 109.786 | 17.500 | 6.7672577 | 10.8–33.5 |
| pytorch-eager-2 | 7,734.215 | 77.342 | 8.543 | 6.7672577 | 8.8–12.4 |
| pytorch-eager-3 | 7,883.785 | 78.838 | 8.976 | 6.7672577 | 6.3–11.4 |
| pytorch-compile-1 | 5,117.727 | 51.177 | 5.915 | 6.7675071 | 22.3–30.8 |
| pytorch-compile-2 | 4,259.148 | 42.591 | 4.614 | 6.7675071 | 10.6–12.3 |
| pytorch-compile-3 | 4,311.663 | 43.117 | 4.945 | 6.7675071 | 10.5–11.7 |

The load rose during `pytorch-eager-1` (10.8 to 33.5) and stayed at 22.3 to 30.8 through
`pytorch-compile-1`; both are the slowest runs of their sides. The two `torch.compile` runs at a
load of 10.5 to 12.3 measured 4,259.1 and 4,311.7 µs; ModelSpec's three runs, at 7.1 to 22.3,
measured 4,118.9 to 4,163.9, a spread of 1.1 %. `modelspec-2` began at a load of 20.9
(`uptime.txt`) and measured 4,130.3. The record keeps every run as measured; no run was dropped
or repeated.

ModelSpec's training seconds to the target do not follow its µs per step: run 3 reached step
1,100 after 4.663 s (4,239 µs per step up to there) and run 1 after 4.058 s (3,689). The cause is
not investigated.

## Held-out loss curves

ModelSpec's 101 held-out losses are identical across its 3 runs and within 7e-7 of the
2026-10-29 runs'. The PyTorch runs agree with each other within 9.6e-9 per mode and with
2026-10-29's eager within 9.5e-9. Between sides: |ModelSpec − eager| at most 2.21e-3 nats,
|ModelSpec − compile| 1.09e-3, |eager − compile| 2.62e-3.

| step | ModelSpec | eager | compile |
|---|---|---|---|
| 0 | 9.029766 | 9.029767 | 9.029767 |
| 1,000 | 6.928237 | 6.928291 | 6.928193 |
| 1,100 | 6.923570 | 6.923565 | 6.923509 |
| 5,000 | 6.846160 | 6.845315 | 6.845205 |
| 10,000 | 6.767498 | 6.767258 | 6.767507 |

## Evidence

The raw records are in `benchmarks/raw/2026-11-05-bert-tiny-pretrain-3/`, laid out as those of
the first series of 2026-11-05: `modelspec-<n>.stdout.txt` (stdout without its `init` and
`param` lines), `modelspec-<n>.jsonl` (the `--record` event files), `pytorch-<mode>-<n>.jsonl`
and `pytorch-<mode>-<n>.stderr.txt` (the venv path replaced by `<venv>`), `loss-modelspec.txt`,
`loss-pytorch.txt`, `loss-pytorch.stderr.txt`, `clocks.csv` (local time, UTC+1), `load.txt`,
`times.txt`, `uptime.txt`, `display.txt`, `run-clocks.txt` (with the gate verdicts),
`commit.txt` (commit, version, dirty flag and binary SHA-256), `driver.txt`,
`pytorch-version.txt` and `gpu-procs-<run>.txt`. The opponent's result lines carry
`"dirty": false`, so the docs test `every_bert_tiny_pretrain_entry_repeats_the_raw_records_it_cites`
admits the three rows of `benchmarks/runs.yaml` that cite them. The series at 5ca80f2e is in
`benchmarks/raw/2026-11-05-bert-tiny-pretrain-2/` in the same layout.

| file | SHA-256 |
|---|---|
| `modelspec-1.jsonl` | `6a7405bd0bc75ba0281f48b10f9b52d61128a02e8112bf5341cd9b8bd8a4ed32` |
| `modelspec-2.jsonl` | `d4c1b60e407dccd9287205026c6b353628e491802970dc209a07ce15558f29ce` |
| `modelspec-3.jsonl` | `820c9095c6ab1ef20f4835949ec65bc8e1c24eed1a7de598d35fdc7778d8d689` |
| `pytorch-eager-1.jsonl` | `64c93b665d17f31afbe56055a2322114d9cc3350172cac47657278a9944bf7da` |
| `pytorch-eager-2.jsonl` | `331e5099dc346cfbf48c867f8d36c119436119199a8703b12981c1b23d14765f` |
| `pytorch-eager-3.jsonl` | `b6516b1f558e39f73e464b925fd950cfd19ee99c1b4b29fa5c1f8d9cddc013ea` |
| `pytorch-compile-1.jsonl` | `b08aee32038fbe8d854241a077bf64888f87553ebf73615fc77e928b99cc510e` |
| `pytorch-compile-2.jsonl` | `43a984d3d0ed11783b7f18fa10cf9610a5e68c9b247ece7bd964f2f6035ef871` |
| `pytorch-compile-3.jsonl` | `456503f02276e8056e91ac8f0047bb635c558f713005f687297b2292bcf77fd9` |
| `initial.msckpt` (not committed, 5,951,104 bytes) | `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704` |
