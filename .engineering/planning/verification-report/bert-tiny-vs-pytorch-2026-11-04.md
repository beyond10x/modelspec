---
format: aep.planning-md/3
id: verification-report:bert-tiny-vs-pytorch-2026-11-04
kind: verification-report
status: draft
title: 'BERT-tiny pre-training on the RTX 3090 after three kernel changes: ModelSpec 5.5 ms/step against PyTorch eager 9.1 ms and compile 4.9 ms, all three at the baseline by step 1,100'
summary: 'Timed runs at acf24d85 plus the uncommitted cross-entropy, long-k and embedding-sort changes, 3 runs per side of 10,000 steps with the GPU in P0 or P2 at 1,700 MHz SM or more: ModelSpec 0.60x eager and 1.13x torch.compile per step, 0.62x and 1.12x to the held-out unigram baseline; every side''s clock excludes the held-out evaluations; the records are of a dirty tree and the rows wait for a clean re-run'
relations:
- reviews: epic:bert-tiny-parity-with-eager
revision: 2
---
# BERT-tiny pre-training on the RTX 3090 after three kernel changes: ModelSpec against PyTorch eager and torch.compile

Reviews `epic:bert-tiny-parity-with-eager` (step E of `initiative:bert-tiny-beats-pytorch`): the
median µs per step of `examples/bert_tiny.mspec` at batch 32 on Vulkan against PyTorch eager's,
measured the same day on the same device with the harness `opponents/bert_tiny_pretrain/`, three
timed runs per side. The build is release 0.25.0 (acf24d85) plus the changes of
`story:vulkan-cross-entropy-row-workgroup`, `story:vulkan-long-k-from-4096` and
`story:vulkan-embedding-sort-lanes`, uncommitted at the time of the runs (below).

**Verdict: ModelSpec's step is 0.60 of eager's and 1.13 of `torch.compile`'s.** Medians of 3
runs of 10,000 steps: ModelSpec Vulkan 5,482.6 µs per step, PyTorch CUDA eager 9,113.8,
`torch.compile` 4,857.7. ModelSpec's slowest run (5,567.0) is under eager's fastest (7,691.1).
All three sides first reach the held-out unigram baseline at step 1,100. The epic's outcome
holds on these records; step F (`epic:bert-tiny-beats-torch-compile`) does not yet.

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

The commands of 2026-10-29. The series ran as one script: before it `xset dpms force on`,
`xset -dpms` and `xset s off`; before each run `xset s reset`, 5 s, and `xset q` must report
`Monitor is On` or `DPMS is Disabled` (`display.txt`: `DPMS is Disabled` before every run); after
it `xset +dpms` and `xset s on`. Each run holds `flock` on `~/.cache/modelspec-gpu.lock`. The
runs alternate ModelSpec, eager, compile, three times. `<packed-128>` is the packed WikiText-103
directory of `story:bert-tiny-gpu-pretraining` (binding `668f3433-dbd7-8871-a5a7-a421c2bd1816`).
A sampler appended the 1-minute load average to `load.txt` every 10 s.

## Host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 615.71.09 (2026-10-29: 610.57.04), clocks not locked |
| ModelSpec | `modelspec 0.25.0` at acf24d8508c0a60ac8d1be937c63144799aa5fa4 plus uncommitted changes, `git diff` SHA-256 `1b3900409e96e2c20521eaed9722a4f81342567075c0d2ae68d93ec647e1cd97`; release binary SHA-256 `6886a4d03f46cc56dd34860848e39408ce102984eaf6875f842fd43546aa03f1` (`commit.txt`) |
| PyTorch | torch 2.15.0+cu132, CUDA 13.2, Python 3.14.7 (2026-10-29: 2.14.1+cu130, 13.0), tf32 off, torch.compile mode default |
| other compute processes | none before each of the 11 GPU runs (`gpu-procs-<run>.txt`, header only) |
| clock samples | 2,339, every 500 ms, 23:46:14 to 00:05:51 local (UTC+1) |
| load samples | 117, every 10 s |

## Comparability

| | ModelSpec Vulkan | PyTorch eager and compile | same |
|---|---|---|---|
| steps | 10,000 (closing line `steps 10000`) | 10,000 (result line `"steps": 10000`) | yes |
| held-out evaluation | step 0 and every 100 steps, 101 `eval` lines per run | 101 evaluations per run (102 JSON lines with the result line) | yes |
| batch and sequence | 32 sequences of 128, 4,096 tokens | the same (result line) | yes |
| data | binding `668f3433-…` | the same directory, ModelSpec's draw | yes |
| initial parameters | the 41 `init` lines of each of the 3 timed runs equal those of the `--steps 0` run (SHA-256 of the block `9c64ca8bcfd789819a2eb691a2d498511a094fc68d576a7c3fd6f7eeb7cd64cf`, as on 2026-10-29) | `initial.msckpt`, SHA-256 `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704`, byte-identical to 2026-10-29's | yes |
| step-0 held-out loss | 9.029766 in all 3 runs and the `--steps 0` run | 9.029766850451333 in all 6 runs; `--loss-check` 9.029767 | yes, within the printed digit |
| target | 6.926537926202018 | 6.926537926202018 | yes |
| FLOP per step | 37,446,746,112, `MatmulOnly` | 37,446,746,112, `MatmulOnly` | yes |
| training clock | per step, from recording the step's command buffer to the return of its host wait (`--clock`); batch draw and staging, plan build and pipeline creation excluded; **the 101 held-out evaluations excluded**, timed into `eval_seconds` (19.3, 19.9, 20.1 s; story:vulkan-stepwise-multi-step-submit, 0.23.0) | per step, from a CUDA synchronize to one after the scheduler's step; evaluations and batch draw excluded | yes (2026-10-29: ModelSpec's included them) |
| GPU clocks, counted runs | SM 210–1965 MHz, median 1890 (n = 464); state P0; memory median 9751 MHz | eager SM 210–1935, median 1890 (n = 956); compile 210–1920, median 1890 (n = 735); state P2; memory median 9501 MHz | each side at its driver's full-power state |
| warm-up before the clock | none | 3 forward and backward passes and one AdamW step on a copy | differs, as before |
| dropout | ModelSpec's Philox stream | torch's own RNG | differs; the held-out curves agree within 2.3e-3 nats (below) |
| host load (1-minute) | 12.0 to 17.7, median 13.3 | eager 11.0 to 29.7, median 18.4; compile 10.8 to 26.9, median 19.2 | differs, per run below |

## The clock gate

After each run, the median SM clock and the most frequent performance state of the clock samples
inside its span (`run-clocks.txt`). A run with a median under 1,700 MHz or a state other than P0
and P2 would be re-run once; none was.

| run | median SM MHz | state | gate |
|---|---|---|---|
| modelspec-1 | 1920 | P0 | passed |
| pytorch-eager-1 | 1890 | P2 | passed |
| pytorch-compile-1 | 1890 | P2 | passed |
| modelspec-2 | 1875 | P0 | passed |
| pytorch-eager-2 | 1905 | P2 | passed |
| pytorch-compile-2 | 1890 | P2 | passed |
| modelspec-3 | 1830 | P0 | passed |
| pytorch-eager-3 | 1890 | P2 | passed |
| pytorch-compile-3 | 1860 | P2 | passed |

## Results (median of 3 runs of 10,000 steps)

| side | µs per step | tokens/s | GFLOP/s | steps to target | training s to target | final held-out loss |
|---|---|---|---|---|---|---|
| ModelSpec Vulkan | 5,482.596 | 747,091 | 6,830.1 | 1,100 | 5.759 | 6.767498 |
| PyTorch CUDA eager | 9,113.771 | 449,430 | 4,108.8 | 1,100 | 9.329 | 6.767258 |
| PyTorch CUDA torch.compile | 4,857.652 | 843,206 | 7,708.8 | 1,100 | 5.133 | 6.767507 |

Ratios of medians: ModelSpec / eager 0.602 per step and 0.617 to the target; ModelSpec /
compile 1.129 per step and 1.122 to the target. Against 2026-10-29 (15,245.2 µs with the
evaluations inside the clock): 0.360; the 0.25.0 build before these three changes profiled at a
steady 6,763 µs per step (`story:vulkan-cross-entropy-row-workgroup`, Context), this build at
5,346 in the same 200-step profile.

## Each run

| run | µs per step | training s | training s to target | final loss | load (1-min) min–max |
|---|---|---|---|---|---|
| modelspec-1 | 5,372.040 | 53.720 | 5.725 | 6.767498 | 12.4–13.6 |
| modelspec-2 | 5,482.596 | 54.826 | 5.782 | 6.767498 | 12.0–14.3 |
| modelspec-3 | 5,567.011 | 55.670 | 5.759 | 6.767498 | 13.5–17.7 |
| pytorch-eager-1 | 7,691.144 | 76.911 | 8.352 | 6.7672577 | 11.0–13.6 |
| pytorch-eager-2 | 10,715.758 | 107.158 | 9.756 | 6.7672577 | 11.0–29.7 |
| pytorch-eager-3 | 9,113.771 | 91.138 | 9.329 | 6.7672577 | 17.4–24.1 |
| pytorch-compile-1 | 4,652.468 | 46.525 | 4.704 | 6.7675071 | 10.8–14.7 |
| pytorch-compile-2 | 4,983.063 | 49.831 | 5.180 | 6.7675071 | 19.0–26.9 |
| pytorch-compile-3 | 4,857.652 | 48.577 | 5.133 | 6.7675071 | 17.5–24.0 |

The host was loaded by other sessions throughout (another agent's builds; `cargo clean` of this
tree's dev profile ran during `modelspec-1` and `pytorch-eager-1`). Eager's per-step time moved
with the load (7,691 at 11.0–13.6, 10,716 at up to 29.7, 9,114 at 17.4–24.1): its step is bound
by the host's per-operator launches. ModelSpec's three runs spread 3.6 % and compile's 7.1 %. On
2026-10-29 eager measured 7,806.5 at load 2.6–15.5. The record keeps the figures as measured;
even eager's run 1 is 1.38× ModelSpec's slowest run.

## Held-out loss curves

ModelSpec's 101 held-out losses are identical across its 3 runs and within 3e-7 of the
2026-10-29 runs' (the three changes are in the tolerance class of DESIGN.md § 5.4; the CPU's
bits are unchanged). The PyTorch runs agree with each other within 1e-8 per mode and with
2026-10-29's eager within 1.2e-8. Between sides: |ModelSpec − eager| at most 2.21e-3 nats,
|ModelSpec − compile| 1.09e-3, |eager − compile| 2.62e-3.

| step | ModelSpec | eager | compile |
|---|---|---|---|
| 0 | 9.029766 | 9.029767 | 9.029767 |
| 1,000 | 6.928237 | 6.928291 | 6.928193 |
| 1,100 | 6.923570 | 6.923565 | 6.923509 |
| 5,000 | 6.846160 | 6.845315 | 6.845205 |
| 10,000 | 6.767498 | 6.767258 | 6.767507 |

## Evidence

The raw records are in `benchmarks/raw/2026-11-04-bert-tiny-pretrain/`, laid out as those of
2026-10-29: `modelspec-<n>.stdout.txt` (stdout without its `init` and `param` lines),
`modelspec-<n>.jsonl` (the `--record` event files), `pytorch-<mode>-<n>.jsonl` and
`pytorch-<mode>-<n>.stderr.txt` (the venv path replaced by `<venv>`), `loss-modelspec.txt`,
`loss-pytorch.txt`, `loss-pytorch.stderr.txt`, `clocks.csv` (local time, UTC+1), `load.txt`,
`times.txt`, `uptime.txt`, `display.txt`, `run-clocks.txt` (with the gate verdicts),
`commit.txt` (commit, version, dirty flag, diff and binary SHA-256), `driver.txt`,
`pytorch-version.txt` and `gpu-procs-<run>.txt`.

| file | SHA-256 |
|---|---|
| `modelspec-1.jsonl` | `f6ff542b1418c594ef050e6f3ab220f2c68ce21f57f8ae36976fc8f61553386b` |
| `modelspec-2.jsonl` | `8bc23d8bd4a588438ce6bb33db8ae18324d7ce4f9728973de38ff54f2e2a652f` |
| `modelspec-3.jsonl` | `1101acb46a7371a94c00b05ef3cade0d1aff87787373e0525f5f37c53ca2fc52` |
| `pytorch-eager-1.jsonl` | `25c59559f62943163e10a92b28e7126352f81b5940a616b1d8745b434917a245` |
| `pytorch-eager-2.jsonl` | `74df0f2f3cdf0ff530bc78f2376de357e96e6e2c7a07fa82bc7fa9e9e18d23b1` |
| `pytorch-eager-3.jsonl` | `212f9fb10c73e1c29f91c5636843327abd28d49c2e98fdc1e380047f6c861ed9` |
| `pytorch-compile-1.jsonl` | `236e29e1b6571a146733691acf44f0b6391050a51edfd4344a5a49d403d17764` |
| `pytorch-compile-2.jsonl` | `43799b63c3a019f9c99271e90de564d7fb9f26fc22add429bca3fb7718486b7f` |
| `pytorch-compile-3.jsonl` | `7edcb8322f6b3af5efe106e2e1fd42d67453cb147a8ef6b8d0f81c5d58ac0fbd` |
| `initial.msckpt` (not committed, 5,951,104 bytes) | `74735a92b6aa039f45ff7b45330143d41721ff808c6b38bb17bb991492fb3704` |

## Caveat: the records are of a dirty tree

The opponent's result lines carry `"dirty": true` and `commit.txt` says `dirty: true`: the three
kernel stories were not committed when the series ran. The docs test
`every_bert_tiny_pretrain_entry_repeats_the_raw_records_it_cites` refuses such records for a row
of `benchmarks/runs.yaml`, so the rows citing this directory do not pass the gate until the series
is re-run on the committed tree and the records replaced. The figures above are the measurement
of this code; the published rows will be the clean re-run's.
