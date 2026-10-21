---
title: What ModelSpec is
slug: /
sidebar_label: Overview
description: A declarative, statically typed language and a Rust compiler for differentiable tensor programs. M0 trains on a CPU reference; GPU execution is the next milestone.
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

ModelSpec is at milestone M0; the latest release is [0.1.1](https://github.com/beyond10x/modelspec/releases/tag/0.1.1).

| Works in 0.1.1 | Not built yet |
|---|---|
| `modelspec check`: parse and check names, types and shapes | any GPU execution: there is no Vulkan backend |
| `modelspec ir --stage tensor` and `--stage training`: print the verified IR | `compile`, `infer`, `inspect-device`, `bench` |
| `modelspec train --backend cpu`: train on the CPU reference with seeded synthetic data | checkpoints, real datasets |
| one `model` and one `train` block per file, static `f32` shapes, nine operators, SGD | modules, layers, generics, other dtypes and optimizers |

`modelspec train --backend vulkan` is refused with `E-TARGET-001` until M1. [Status](./status.mdx)
lists every shipped capability with the test that holds it, and [Roadmap](./roadmap.md) says what
M1 to M4 add.

## What it is for

- **Errors before anything runs.** A `matmul` whose operands do not fit is a compile error naming
  both types and the dimension it expected ([a shape error](./examples/shape-error.md)), not an
  exception after the first batch.
- **The training step is compiled.** Forward pass, loss, backward pass and parameter update are
  one program the compiler can see whole, so later milestones can fuse and schedule across all of
  it ([the pipeline](./concepts/pipeline.md)).
- **Correctness before speed.** The CPU interpreter is the oracle. Every gradient rule is checked
  against finite differences, and a GPU backend will be held to the CPU reference within stated
  tolerances.

## Where to go next

- [Getting started](./getting-started.md): download the 0.1.1 binary and train the example.
- [The language](./concepts/language.md): what an M0 file may contain.
- [Write a model](./guides/write-a-model.md): build a two-layer model step by step.
- [Pitch deck](pathname:///modelspec/pitch/): the case for ModelSpec in slides.
- [DESIGN.md](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md): the design
  specification, the authority for everything on this site.
