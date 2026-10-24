---
title: What ModelSpec is
slug: /
sidebar_label: Overview
description: A declarative, statically typed language and a Rust compiler for differentiable tensor programs. It trains on a CPU reference and on one Vulkan GPU; GPU execution, M1, is in progress.
---

# What ModelSpec is

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

ModelSpec is in milestone M1 (GPU execution); the latest release is [0.9.0](https://github.com/beyond10x/modelspec/releases/tag/0.9.0).

| Works in 0.9.0 | Not built yet |
|---|---|
| `modelspec check`: parse and check names, types and shapes | `bench` |
| `modelspec ir --stage tensor`, `--stage training` and `--stage kernel`: print the verified IR and the kernel plan | more than one device |
| `modelspec train --backend cpu`: train on the CPU reference with seeded synthetic data or dense `f32` tensor files (`--data`) | |
| `modelspec train --backend vulkan [--stats] [--profile]`: train on one Vulkan device with one host wait per run | modules, layers, generics, other dtypes and optimizers |
| `modelspec train --checkpoint`, `--resume` and `modelspec infer --backend cpu\|vulkan`: checkpoints and inference on the CPU or one Vulkan device | |
| `modelspec compile --backend cpu\|vulkan --output <dir>`: a sealed artifact directory | |
| `modelspec inspect-device --backend vulkan`: each device's profile and negotiation verdict | |
| `modelspec train --serve <addr>`: stream the run as typed events over Server-Sent Events | |
| `modelspec train --record <path>` and `modelspec observe <path>`: record a run and replay it | |
| `--ui ui/dist` on `train --serve` or `observe`: the observatory UI in the browser | |

The observatory UI draws the loss and every metric per step, and for a CPU run the norm and
histogram of each parameter's values and gradient on sampled steps (`--statistics-every`), for a
Vulkan run those of its values at the first and the last step.

On the MNIST benchmark example, `train --backend vulkan` measured 27.14 µs per step on an RTX
3090 in 0.6.0 (median of 5 timed runs of 144,000 steps); no new timed step time was measured for
0.7.0 or 0.7.1, and ADR-007's accuracy and final-weight conditions were not measured. Since 0.7.1,
`--profile` prints each dispatch's FLOP, bytes, GFLOP/s and GB/s; on the same example it reached
447.8 GFLOP/s and 56.6 GB/s over 45.925 µs per profiled step. [Status](./status.mdx) lists every shipped
capability with the test that holds it, and [Roadmap](./roadmap.md) says what M1 to M4 add.

## What it is for

- **Errors before anything runs.** A `matmul` whose operands do not fit is a compile error naming
  both types and the dimension it expected ([a shape error](./examples/shape-error.md)), not an
  exception after the first batch.
- **The training step is compiled.** Forward pass, loss, backward pass and parameter update are
  one program the compiler can see whole, so later milestones can fuse and schedule across all of
  it ([the pipeline](./concepts/pipeline.md)).
- **Correctness before speed.** The CPU interpreter is the oracle. Every gradient rule is checked
  against finite differences, and the GPU backend is held to the CPU reference within stated
  tolerances.

## Where to go next

- [Getting started](./getting-started.md): download the 0.9.0 binary and train the examples on the CPU or a GPU.
- [The language](./concepts/language.md): what an M0 file may contain.
- [Write a model](./guides/write-a-model.md): build a two-layer model step by step.
- [Pitch deck](pathname:///modelspec/pitch/): the case for ModelSpec in slides.
- [DESIGN.md](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md): the design
  specification, the authority for everything on this site.
