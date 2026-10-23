# MLP size sweep: the PyTorch CUDA opponent

The PyTorch side of `story:mlp-size-sweep-vs-pytorch-cuda`. ModelSpec's side is the bench
`crates/modelspec-runtime/benches/mlp_sweep.rs`. No ModelSpec build, test, gate or `task` target
runs this directory ([AGENTS.md](../../AGENTS.md), Rules).

## What it measures

The same 12 points as the bench: 784 -> H ReLU -> 10, no bias, `f32`, MSE, SGD at lr 0.001,
hidden 64, 256, 1024, 4096 and batch 100, 1000, 8192, on synthetic teacher-student data made on
the GPU once per point. Each point runs PyTorch eager and `torch.compile` (`dynamic=False`, after a
dynamo reset), CUDA, TF32 off.

Per point and mode: a 20-step warm-up, calibration runs until one lasts at least `--min-seconds`
(default 5), then 3 timed runs of that many steps. A timed run is host wall time between a
`torch.cuda.synchronize(device)` of the `--device` GPU before the first step and one after the
last. A step is
`zero_grad(set_to_none=True)`, the loss, `backward()` and `SGD.step()` (`foreach=True`).

## Output

One JSON line per point and mode on stdout, in the shape the bench writes:

| field | meaning |
|---|---|
| `side` | `pytorch-eager` or `pytorch-compile` |
| `batch`, `hidden` | the point |
| `flop_per_step`, `flop_rule` | `3196·B·H`, `matmul-only`: the five matmuls of a step, no elementwise work |
| `device` | GPU name, torch and CUDA versions, the `torch.compile` mode |
| `build` | `commit`: the git commit of the checkout; `dirty`: whether its tracked files differ from it. The bench adds `version`, the package version |
| `status` | `measured` or `refused` |
| `steps`, `runs`, `median_us_per_step`, `gflops` | measured: steps per timed run; per run `us_per_step`, `seconds` and `span_unix_ms`; the median; `flop_per_step` over the median |
| `refusal` | refused: the exception it raised |

`span_unix_ms` places each run among GPU clock samples taken with `nvidia-smi`.

## Running it

Needs Python 3.10 or later and a CUDA build of PyTorch, in a virtual environment outside the
repository, on a GPU with no other compute process:

```console
python3 -m venv ~/.cache/modelspec-opponents-venv
~/.cache/modelspec-opponents-venv/bin/pip install torch
~/.cache/modelspec-opponents-venv/bin/python opponents/mlp_sweep/mlp_sweep.py > pytorch.jsonl
```

`--point BATCH:HIDDEN` (repeatable) runs a subset, `--modes eager` or `--modes compile` one mode.
The bench's `--crossover` reads its output beside the bench's:

```console
cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep -- --crossover modelspec.jsonl pytorch.jsonl
```
