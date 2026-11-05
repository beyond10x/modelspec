---
format: aep.planning-md/3
id: verification-report:bert-tiny-vs-pytorch-2026-11-05
kind: verification-report
status: draft
title: 'BERT-tiny pre-training on the RTX 3090 at 30247795: ModelSpec 5.6 ms/step against PyTorch eager 10.2 ms and compile 5.7 ms, all three at the baseline by step 1,100'
summary: 'Clean re-run of the 2026-11-04 series at commit 30247795, 3 runs per side of 10,000 steps with the GPU in P0 or P2 at 1,700 MHz SM or more: ModelSpec 0.55x eager and 0.99x torch.compile per step, 0.62x and 0.94x to the held-out unigram baseline; every side''s clock excludes the held-out evaluations; loaded host (1-minute load 18.7 to 55.7), so the compile figure is within noise and not claimed'
relations:
- reviews: epic:bert-tiny-parity-with-eager
revision: 2
---
# BERT-tiny pre-training on the RTX 3090 at 30247795: ModelSpec against PyTorch eager and torch.compile

Reviews `epic:bert-tiny-parity-with-eager` (step E of `initiative:bert-tiny-beats-pytorch`): the
median µs per step of `examples/bert_tiny.mspec` at batch 32 on Vulkan against PyTorch eager's,
measured the same day on the same device with the harness `opponents/bert_tiny_pretrain/`, three
timed runs per side. The build is commit 30247795 on the branch `perf/beats-pytorch`: release
0.25.0 (acf24d85) plus the committed changes of `story:vulkan-cross-entropy-row-workgroup`,
`story:vulkan-long-k-from-4096` and `story:vulkan-embedding-sort-lanes`. This is the clean
re-run of `verification-report:bert-tiny-vs-pytorch-2026-11-04`, whose records were of the same
binary (SHA-256 below) built from the same changes before they were committed.

**Verdict: ModelSpec's step is 0.55 of eager's and 0.99 of `torch.compile`'s.** Medians of 3
runs of 10,000 steps: ModelSpec Vulkan 5,619.4 µs per step, PyTorch CUDA eager 10,169.5,
`torch.compile` 5,663.6. ModelSpec's slowest run (5,672.1) is under eager's fastest (9,660.5).
All three sides first reach the held-out unigram baseline at step 1,100. The epic's outcome
holds on these records. The figure against `torch.compile` is not a lead for step F
(`epic:bert-tiny-beats-torch-compile`): the host was loaded throughout (below), and the same
binary measured 5,482.6 against `torch.compile`'s 4,857.7 on 2026-11-04 at lower load.

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

The commands and script of 2026-11-04: before the series `xset dpms force on`, `xset -dpms`
and `xset s off`; before each run `xset s reset`, 5 s, and `xset q` must report `Monitor is On`
or `DPMS is Disabled` (`display.txt`: `DPMS is Disabled` before every run); after it
`xset +dpms` and `xset s on`. Each run holds `flock` on `~/.cache/modelspec-gpu.lock`. The runs
alternate ModelSpec, eager, compile, three times. `<packed-128>` is the packed WikiText-103
directory of `story:bert-tiny-gpu-pretraining` (binding `668f3433-dbd7-8871-a5a7-a421c2bd1816`).
A sampler appended the 1-minute load average to `load.txt` every 10 s.

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 615.71.09, clocks not locked |
| ModelSpec | `modelspec 0.25.0` at 30247795da4c44b5ee129d29d61498ed1e427100, `dirty: false`; release binary SHA-256 `6886a4d03f46cc56dd34860848e39408ce102984eaf6875f842fd43546aa03f1` (`commit.txt`), the binary of 2026-11-04 |
| PyTorch | torch 2.15.0+cu132, CUDA 13.2, Python 3.14.7, tf32 off, torch.compile mode default |
| other compute processes | none before each of the 11 GPU runs (`gpu-procs-<run>.txt`, header only) |
| clock samples | 2,608, every 500 ms, 00:34:08 to 00:55:55 local (UTC+1) on 2026-11-05 |
| load samples | 131, every 10 s |

## Comparability

| | ModelSpec Vulkan | PyTorch eager and compile | same |
|---|---|---|---|
| steps | 10,000 (closing line `steps 10000`) | 10,000 (result line `"steps": 10000`) | yes |
| held-out evaluation | step 0 and every 100 steps, 101 `eval` lines per run | 101 evaluations per run (102 JSON lines with the result line) | yes |
| batch and sequence | 32 sequences of 128, 4,096 tokens | the same (result line) | yes |
| data | binding `668f3433-…` | the same directory, ModelSpec's draw | yes |
| initial parameters | the 41 `init` lines of each of the 3 timed runs equal those of the `--steps 0` run (SHA-256 of the block `9c64ca8bcfd789819a2eb691a2d498511a094fc68d576a7c3fd6f7eeb7cd64cf`, as on 2026-10-29 and 2026-11-04) | `initial.msckpt`, SHA-256 `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704`, byte-identical to 2026-10-29's | yes |
| step-0 held-out loss | 9.029766 in all 3 runs and the `--steps 0` run | 9.029766850451333 in all 6 runs; `--loss-check` 9.029767 | yes, within the printed digit |
| target | 6.926537926202018 | 6.926537926202018 | yes |
| FLOP per step | 37,446,746,112, `MatmulOnly` | 37,446,746,112, `MatmulOnly` | yes |
| training clock | per step, from recording the step's command buffer to the return of its host wait (`--clock`); batch draw and staging, plan build and pipeline creation excluded; the 101 held-out evaluations excluded, timed into `eval_seconds` (19.6, 20.0, 20.3 s; story:vulkan-stepwise-multi-step-submit, 0.23.0) | per step, from a CUDA synchronize to one after the scheduler's step; evaluations and batch draw excluded | yes |
| GPU clocks, counted runs | SM 210–1935 MHz, median 1860 (n = 477); state P0; memory median 9751 MHz | eager SM 210–1935, median 1890 (n = 1091); compile 210–1935, median 1890 (n = 835); state P2; memory median 9501 MHz | each side at its driver's full-power state |
| warm-up before the clock | none | 3 forward and backward passes and one AdamW step on a copy | differs, as before |
| dropout | ModelSpec's Philox stream | torch's own RNG | differs; the held-out curves agree within 2.3e-3 nats (below) |
| host load (1-minute) | 21.5 to 45.2, median 27.6 | eager 18.7 to 55.7, median 30.0; compile 23.6 to 47.2, median 29.5 | differs, per run below |

## The clock gate

After each run, the median SM clock and the most frequent performance state of the clock samples
inside its span (`run-clocks.txt`). A run with a median under 1,700 MHz or a state other than P0
and P2 would be re-run once; none was.

| run | median SM MHz | state | gate |
|---|---|---|---|
| modelspec-1 | 1905 | P0 | passed |
| pytorch-eager-1 | 1905 | P2 | passed |
| pytorch-compile-1 | 1890 | P2 | passed |
| modelspec-2 | 1860 | P0 | passed |
| pytorch-eager-2 | 1890 | P2 | passed |
| pytorch-compile-2 | 1890 | P2 | passed |
| modelspec-3 | 1800 | P0 | passed |
| pytorch-eager-3 | 1890 | P2 | passed |
| pytorch-compile-3 | 1890 | P2 | passed |

## Results (median of 3 runs of 10,000 steps, each figure its own median)

| side | µs per step | tokens/s | GFLOP/s | steps to target | training s to target | final held-out loss |
|---|---|---|---|---|---|---|
| ModelSpec Vulkan | 5,619.427 | 728,900 | 6,663.8 | 1,100 | 5.956 | 6.767498 |
| PyTorch CUDA eager | 10,169.538 | 402,772 | 3,682.2 | 1,100 | 9.680 | 6.767258 |
| PyTorch CUDA torch.compile | 5,663.632 | 723,211 | 6,611.8 | 1,100 | 6.357 | 6.767507 |

Ratios of medians: ModelSpec / eager 0.553 per step and 0.615 to the target; ModelSpec /
compile 0.992 per step and 0.937 to the target. Against 2026-10-29 (15,245.2 µs with the
evaluations inside the clock): 0.369. Against 2026-11-04, the same binary from the dirty tree:
ModelSpec 1.025, eager 1.116, compile 1.166; the 2026-11-04 series ran at 1-minute load 10.8 to
29.7, this one at 18.7 to 55.7.

## Each run

| run | µs per step | training s | training s to target | final loss | load (1-min) min–max |
|---|---|---|---|---|---|
| modelspec-1 | 5,461.680 | 54.617 | 5.868 | 6.767498 | 21.5–27.7 |
| modelspec-2 | 5,619.427 | 56.194 | 5.956 | 6.767498 | 27.0–45.2 |
| modelspec-3 | 5,672.129 | 56.721 | 5.980 | 6.767498 | 22.0–39.0 |
| pytorch-eager-1 | 9,660.498 | 96.605 | 8.924 | 6.7672577 | 18.7–29.4 |
| pytorch-eager-2 | 10,468.710 | 104.687 | 9.680 | 6.7672577 | 41.2–55.7 |
| pytorch-eager-3 | 10,169.538 | 101.695 | 10.791 | 6.7672577 | 20.5–30.9 |
| pytorch-compile-1 | 5,663.632 | 56.636 | 6.357 | 6.7675071 | 25.4–30.0 |
| pytorch-compile-2 | 5,654.525 | 56.545 | 5.895 | 6.7675071 | 41.7–47.2 |
| pytorch-compile-3 | 5,967.823 | 59.678 | 8.527 | 6.7675071 | 23.6–31.9 |

The host was loaded by other sessions throughout (1-minute load 18.7 to 55.7; 2026-11-04: 10.8
to 29.7; 2026-10-29: 2.6 to 18.7). Both PyTorch sides moved with it: eager measured 9,660 to
10,469 here against 7,691 to 10,716 on 2026-11-04, compile 5,655 to 5,968 against 4,652 to
4,983. ModelSpec's three runs spread 3.9 % (5,462 to 5,672) against 5,372 to 5,567 on
2026-11-04. The record keeps the figures as measured. Eager's fastest run is 1.70× ModelSpec's
slowest; compile's fastest run (5,654.5) is above ModelSpec's median and below its slowest, so
the compile comparison of this series is within its noise and is not claimed either way.

## Held-out loss curves

ModelSpec's 101 held-out losses are identical across its 3 runs and within 3e-7 of the
2026-10-29 runs' (the three changes are in the tolerance class of DESIGN.md § 5.4; the CPU's
bits are unchanged). The PyTorch runs agree with each other within 2.2e-8 per mode and with
2026-10-29's eager within 1.1e-8. Between sides: |ModelSpec − eager| at most 2.21e-3 nats,
|ModelSpec − compile| 1.09e-3, |eager − compile| 2.62e-3.

| step | ModelSpec | eager | compile |
|---|---|---|---|
| 0 | 9.029766 | 9.029767 | 9.029767 |
| 1,000 | 6.928237 | 6.928291 | 6.928193 |
| 1,100 | 6.923570 | 6.923565 | 6.923509 |
| 5,000 | 6.846160 | 6.845315 | 6.845205 |
| 10,000 | 6.767498 | 6.767258 | 6.767507 |

## Evidence

The raw records are in `benchmarks/raw/2026-11-05-bert-tiny-pretrain/`, laid out as those of
2026-10-29 and 2026-11-04: `modelspec-<n>.stdout.txt` (stdout without its `init` and `param`
lines), `modelspec-<n>.jsonl` (the `--record` event files), `pytorch-<mode>-<n>.jsonl` and
`pytorch-<mode>-<n>.stderr.txt` (the venv path replaced by `<venv>`), `loss-modelspec.txt`,
`loss-pytorch.txt`, `loss-pytorch.stderr.txt`, `clocks.csv` (local time, UTC+1), `load.txt`,
`times.txt`, `uptime.txt`, `display.txt`, `run-clocks.txt` (with the gate verdicts),
`commit.txt` (commit, version, dirty flag and binary SHA-256), `driver.txt`,
`pytorch-version.txt` and `gpu-procs-<run>.txt`. The opponent's result lines carry
`"dirty": false`, so the docs test `every_bert_tiny_pretrain_entry_repeats_the_raw_records_it_cites`
admits the three rows of `benchmarks/runs.yaml` that cite them.

| file | SHA-256 |
|---|---|
| `modelspec-1.jsonl` | `6f1afeb05ccfd433c09dde5b42827c5a844911d3737451290f2a4e9843f5cd37` |
| `modelspec-2.jsonl` | `af08207058fb4e9e2e3004de0211c4033ce1e2069c69ab51971a496abe825678` |
| `modelspec-3.jsonl` | `d3e96b10f89623db03195855aa004162bf6c3411eb4599b4e2cfdb222ffa8798` |
| `pytorch-eager-1.jsonl` | `b0638d2c59b94590169b5d5b39183e610c6f7a48317d1ee42448f3dee122ffae` |
| `pytorch-eager-2.jsonl` | `676a5721aa90b6f0103887409f79136431714a77ada1023ba3ad4de516d62777` |
| `pytorch-eager-3.jsonl` | `02070c09c63869182f2d6fec5a3c170a2b80746548e36ef018bf8c9f07eb230b` |
| `pytorch-compile-1.jsonl` | `0e06adfa7225f8b7f20b592fd62fce137312bb7a8bf70adc942ba60cad028037` |
| `pytorch-compile-2.jsonl` | `15dc57bde89c29b7e8ef8e500faf5a2c78793b0ab9ea0ace6bf4a18a80f48a76` |
| `pytorch-compile-3.jsonl` | `7330d803a98340caf5068a534ae5375e5e5bdfcb65297898bfb7646442fa732e` |
| `initial.msckpt` (not committed, 5,951,104 bytes) | `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704` |
