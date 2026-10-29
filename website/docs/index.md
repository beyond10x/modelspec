---
title: What ModelSpec is
slug: /
sidebar_label: Overview
description: ModelSpec is for training models faster than a general-purpose framework, by compiling one declared training program into a runtime built for one target (planned). It is a typed language and a Rust compiler that trains today on a CPU reference and on one Vulkan GPU.
---

# What ModelSpec is

ModelSpec is for training models faster than a general-purpose framework can. The whole training
step is one declared, typed program, so the compiler knows everything before it runs and can build
a runtime for one target: one use case, one operating system, one GPU chipset. Every optimization
is checked against a CPU reference, so a change that alters a result is refused and only faster
ones are kept ([Why ModelSpec](./concepts/north-star.md)).

That speed is planned, not shipped. Today ModelSpec trains on one RTX 3090 through Vulkan, and on
a transformer it is still many times slower than PyTorch: the BERT-tiny encoder block at batch 8
takes 93,868.4 µs per step against 3,937.2 µs on PyTorch eager. Closing that gap on BERT-tiny,
step by step, is the [roadmap](./roadmap.md).

ModelSpec is a declarative, statically typed language for differentiable tensor programs, and a
compiler for it written in Rust. A `.mspec` file says what a model computes, what loss it is
trained on and with which optimizer. The compiler decides how that runs: it checks every name,
type and shape, lowers the file to a typed tensor IR, derives the gradients, adds the optimizer
update and trains the whole step as one program.

There is no Python, PyTorch or CUDA anywhere in the stack. The GPU target is Vulkan compute and
SPIR-V through the system Vulkan loader, so the driver, not a CUDA toolkit, compiles the kernels
for the card.

```modelspec
model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
```

## What works today

ModelSpec is in milestone M1 (GPU execution), and its current goal is a small BERT pre-trained
from scratch; the latest release is [0.25.0](https://github.com/beyond10x/modelspec/releases/tag/0.25.0).
BERT-tiny pre-trained on one RTX 3090 for 10,000 steps reached a held-out loss of 6.7674994 nats,
0.159 below the unigram baseline of 6.9265379 ([the run record](./runs/bert-tiny-gpu-pretraining.md)).

| Works in 0.25.0 | Not built yet |
|---|---|
| `modelspec check`: parse and check names, types and shapes | `bench` |
| `modelspec ir --stage tensor`, `--stage training` and `--stage kernel`: print the verified IR and the kernel plan | more than one device |
| `modelspec train --backend cpu`: train on the CPU reference with seeded synthetic data or dense `f32` and `i32` tensor files (`--data`) | modules, layers, generics and `f16` |
| `modelspec train --backend vulkan [--stats] [--profile]`: train on one Vulkan device with one host wait per run | |
| `modelspec train --backend vulkan --data <packed-dir> [--eval-every <n>]`: train the hidden-32 BERT twin with dropout and AdamW on one Vulkan device, one host wait per segment of up to 8 steps, and evaluate the held-out split on the device in a command buffer of its own, one more host wait per evaluation | |
| `embedding`, `add_rows`, `gelu`, `layer_norm`, `softmax`, `attention`, `cross_entropy` and `dropout`, with `i32` row ids, on the CPU reference; all eight also have Vulkan kernels | |
| `modelspec data prepare`, `data tokenizer` and `data pack`: split a corpus, train a WordPiece tokenizer and pack token sequences | |
| `modelspec train --backend cpu --data <packed-dir> [--eval-every <n>]`: train a masked-language model on packed sequences and evaluate the held-out split during the run | |
| `optimizer = adamw(...)`: AdamW with decoupled weight decay and a linear warm-up and decay of the learning rate, on the CPU reference and one Vulkan device | checkpoints of an AdamW run |
| `optimizer = muon(...)`: orthogonalized momentum for the matrix parameters and AdamW for the rest, on the CPU reference | Muon on a Vulkan device, and checkpoints of a Muon run |
| `modelspec train --checkpoint`, `--resume` and `modelspec infer --backend cpu\|vulkan`: checkpoints and inference on the CPU or one Vulkan device | |
| `modelspec compile --backend cpu\|vulkan --output <dir>`: a sealed artifact directory | |
| `modelspec inspect-device --backend vulkan`: each device's profile and negotiation verdict | |
| `modelspec train --serve <addr>`: stream the run as typed events over Server-Sent Events | |
| `modelspec train --record <path>` and `modelspec observe <path>`: record a run and replay it | |
| `--ui ui/dist` on `train --serve` or `observe`: the observatory UI in the browser | |

The observatory UI draws the loss and every metric per step, and for a CPU run the norm and
histogram of each parameter's values and gradient on sampled steps (`--statistics-every`), for a
Vulkan run those of its values at the first and the last step.

Since 0.11.0, `cross_entropy` and `dropout` run on the CPU reference, `train --backend cpu --data`
on a packed directory trains a masked-language model on the sequences `data pack` wrote, and
`--eval-every` evaluates the held-out split during the run
([Prepare training data](./guides/prepare-training-data.md)).

On the MNIST benchmark example, `train --backend vulkan` measured 27.14 µs per step on an RTX
3090 in 0.6.0 (median of 5 timed runs of 144,000 steps), and ADR-007's accuracy and final-weight
conditions were not measured. Since 0.7.1, `--profile` prints each dispatch's FLOP, bytes,
GFLOP/s and GB/s; on the same example it reached 447.8 GFLOP/s and 56.6 GB/s over 45.925 µs per
profiled step. Those are the last timings through `train`. The sweep harness `mlp_sweep` (since
0.7.1) trains the same shapes, 784 → 64 → 10 at batch 100, on the same GPU: 33.044 µs per step
in 0.9.0 and 30.726 µs in 0.10.0, each the median of 3 timed runs (`benchmarks/runs.yaml`).
[Benchmarks](/docs/benchmarks) charts every run, [Status](/docs/status) lists every shipped
capability with the test that holds it, and [Roadmap](./roadmap.md) says what M1 to M4 add.

## What else it gives you

- **Errors before anything runs.** A `matmul` whose operands do not fit is a compile error naming
  both types and the dimension it expected ([a shape error](./examples/shape-error.md)), not an
  exception after the first batch.
- **The training step is compiled.** Forward pass, loss, backward pass and parameter update are
  one program the compiler can see whole, so it fuses and schedules across all of
  it ([the pipeline](./concepts/pipeline.md)).
- **Correctness before speed.** The CPU interpreter is the oracle. Every gradient rule is checked
  against finite differences, and the GPU backend is held to the CPU reference within stated
  tolerances.

## Where to go next

- [Why ModelSpec](./concepts/north-star.md): the north star, the BERT-tiny ladder against PyTorch,
  and what is shipped and what is planned.
- [Getting started](./getting-started.md): download the 0.25.0 binary and train the examples on the CPU or a GPU.
- [The language](./concepts/language.md): what a file may contain.
- [Write a model](./guides/write-a-model.md): build a two-layer model step by step.
- [Pitch deck](pathname:///modelspec/pitch/): the case for ModelSpec in slides.
- [DESIGN.md](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md): the design
  specification, the authority for everything on this site.
