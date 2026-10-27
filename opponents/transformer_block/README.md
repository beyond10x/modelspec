# Transformer block: the PyTorch CUDA opponent

The PyTorch side of `story:transformer-block-benchmark`. ModelSpec's side is the bench
`crates/modelspec-runtime/benches/transformer_block.rs`, which runs
`examples/transformer_block.mspec`. No ModelSpec build, test, gate or `task` target runs this
directory ([AGENTS.md](../../AGENTS.md), Rules).

## What it measures

The model of `examples/transformer_block.mspec`: two post-layer-norm encoder layers of BERT-tiny
(hidden 128, 2 heads, feed-forward 512, sequences of 128, no dropout), the mean squared error of
the second layer's output against a target of the same shape, `f32`, SGD at lr 0.01, at batches of
8, 32 and 128 sequences. Every operation is written out with `torch` and
`torch.nn.functional` as ModelSpec defines it (DESIGN.md § 2.1): weights `[in, out]` multiplied
from the right, `layer_norm` with `eps` 1e-12, the tanh-form GELU evaluated as
`x / (1 + exp(-2u))`, and attention split into 2 heads of 64 columns, scaled by `1 / sqrt(64)`,
each query attending to the keys of its own sequence whose mask entry is 1. The batch is synthetic
and made on the GPU once per point: `x` standard normal, the key mask all ones, and the target the
prediction of a teacher of the same shapes. The parameters are torch's own Xavier-uniform draw, or
those of a ModelSpec checkpoint with `--checkpoint`. Each point runs PyTorch eager and
`torch.compile` (`dynamic=False`, after a dynamo reset), CUDA, TF32 off.

Per point and mode: a 20-step warm-up, calibration runs until one lasts at least `--min-seconds`
(default 5), then 3 timed runs of that many steps. A timed run is host wall time between a
`torch.cuda.synchronize(device)` of the `--device` GPU before the first step and one after the
last. A step is `zero_grad(set_to_none=True)`, the loss, `backward()` and `SGD.step()`
(`foreach=True`).

## Output

One JSON line per point and mode on stdout, in the shape the bench writes:

| field | meaning |
|---|---|
| `side` | `pytorch-eager` or `pytorch-compile` |
| `batch`, `seq` | the point: sequences per step and their length, 128 |
| `tokens_per_step` | `batch · seq` |
| `flop_per_step`, `flop_rule` | `N·(42·H² + 24·H·F + 24·S·H)` with `N = B·128`, that is `339,738,624·B`, `matmul-only`: the matrix products of a step, forward and backward, no elementwise, layer-norm or softmax work (`benches/transformer_block/block.rs`) |
| `device` | GPU name, torch and CUDA versions, the `torch.compile` mode |
| `build` | `commit`: the git commit of the checkout; `dirty`: whether its tracked files differ from it. The bench adds `version`, the package version |
| `status` | `measured` or `refused` |
| `steps`, `runs`, `median_us_per_step`, `tokens_per_s`, `gflops` | measured: steps per timed run; per run `us_per_step`, `seconds` and `span_unix_ms`; the median; `tokens_per_step` and `flop_per_step` over the median |
| `refusal` | refused: the exception it raised |

`span_unix_ms` places each run among GPU clock samples taken with `nvidia-smi`.

## Running it

Needs Python 3.10 or later and a CUDA build of PyTorch, in a virtual environment outside the
repository, on a GPU with no other compute process. NumPy is not needed:

```console
python3 -m venv ~/.cache/modelspec-opponents-venv
~/.cache/modelspec-opponents-venv/bin/pip install torch
~/.cache/modelspec-opponents-venv/bin/python opponents/transformer_block/transformer_block.py > pytorch.jsonl
```

`--point BATCH` (repeatable) runs a subset, `--modes eager` or `--modes compile` one mode.
`--checkpoint PATH.msckpt` starts every point from the parameters of a ModelSpec checkpoint.
`--profile` runs the 3 timed runs under `torch.profiler` (CUDA activity only) and prints to stderr
one `profile kernel` line per CUDA kernel, memset or copy, the most device time first, then
`profile steps <n> device_us_per_step <µs>`; the timed figures of a profiled run carry the
profiler's cost and are not benchmark figures.

## The first-step loss check

`--loss-check PATH.msckpt --batch DIR` trains nothing: it prints `step 0 loss <v>`, the loss of the
checkpoint's parameters on the batch ModelSpec drew, in the line ModelSpec prints. It is a sanity
check: it shows the opponent reads the checkpoint and the batch, not that the two sides compute
the same function. A GELU of another form moves the loss by less than the run tolerance (below).
ModelSpec writes both inputs with `train --steps 0`: the checkpoint of its initial parameters, and
with `--write-batch DIR` its synthetic batch, one file per input and target declaration,
`DIR/x.f32`, `DIR/key_mask.f32` and `DIR/y.f32`, each the tensor's elements in row-major order as
little-endian `f32` and nothing else:

```console
modelspec train examples/transformer_block.mspec --backend cpu --steps 0 --checkpoint initial.msckpt --write-batch batch
~/.cache/modelspec-opponents-venv/bin/python opponents/transformer_block/transformer_block.py --loss-check initial.msckpt --batch batch
```

The two losses agree within the run tolerance of DESIGN.md § 9, `|g − c| ≤ 1e-4 + 1e-3·|c|`.

## The one-step comparison

`--compare-step STEP1.msckpt --checkpoint initial.msckpt --batch DIR` is the check that the two
sides compute the same function. It takes one SGD step from the initial checkpoint on ModelSpec's
batch and compares the result with ModelSpec's own checkpoint of that step, written from the same
seed:

```console
modelspec train examples/transformer_block.mspec --backend cpu --steps 1 --checkpoint step1.msckpt
~/.cache/modelspec-opponents-venv/bin/python opponents/transformer_block/transformer_block.py --compare-step step1.msckpt --checkpoint initial.msckpt --batch batch
```

It prints the step's loss, one `compare <name>` line per parameter and a summary line, and exits 1
when either check fails:

- The run check: every element of the stepped parameters within the run tolerance of § 9. At lr
  0.01 one step moves a parameter by about `1e-4` of its value, so this check catches a broken
  step, not a changed function.
- The update check: ModelSpec's update `c − initial` is its `−lr·g` rounded once to the `f32` grid
  of `c`. Per element, what half an ulp of `c` does not explain of its difference from the
  opponent's `−lr·g`, taken in `f64`, is the excess. Per parameter, the largest excess over the
  parameter's largest update (at least `1e-2` of the largest update of any parameter, so the key
  biases, whose gradient is zero by construction, are not judged on their rounding noise) must be
  at most `2e-5`.

Measured on 2026-10-27 on an RTX 3090 against `train --backend cpu` at the example's batch of 8
sequences:

| Opponent | `step 0 loss` | Largest run-check difference | Elements outside § 9 | Update excess | Exit |
|---|---|---|---|---|---|
| as written | 0.74829835 | 1.490e-08 | 0 | 1.139e-06 | 0 |
| GELU `0.5·x·(1 + erf(x/√2))` | 0.7483432 | 1.192e-07 | 0 | 4.696e-04 | 1 |
| GELU cubic coefficient 0.04 for 0.044715 | 0.74788797 | 3.576e-07 | 0 | 3.663e-03 | 1 |

ModelSpec printed `step 0 loss 0.74829835`. Both changed functions pass the loss check and the run
check and fail the update check, 23 and 183 times its bound; the opponent as written stays 17 times
under it.

## The checkpoint reader

The reader checks the header's magic, version and endian marker, the header checksum (ModelSpec
refuses a mismatch with E-CKPT-001), the file size the manifest states (E-CKPT-002) and the BLAKE3
hash of the payload (E-CKPT-004). It refuses a file that fails one with a line naming the
condition and ModelSpec's code: `--loss-check` and `--compare-step` exit 1, a timed point is
refused. The hash is computed in pure Python, about 2 s per checkpoint of the example.
