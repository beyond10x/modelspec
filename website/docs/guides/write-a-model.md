---
title: Write a model
description: Build a two-layer model with a relu in the M0 subset, step by step, checking it after each step and training it on the CPU reference.
---

# Write a model

This guide builds a model with two weight matrices and a `relu` between them, in the M0 subset of
the language. The finished file is the example [Two layers with relu](../examples/two-layer-relu.md),
whose page shows everything the command line prints for it. You need the `modelspec` binary
([Getting started](../getting-started.md)): the release has a prebuilt one for Linux x86_64 only,
and on other systems you build it from source.

## 1. The model and its parameters

Create `two_layer.mspec`. The model takes a batch of 16 rows of 4 features and returns one value
per row, through a hidden layer of 8:

```modelspec
model TwoLayer {
    param w1: Tensor<f32, [4, 8]> = xavier_uniform();
    param w2: Tensor<f32, [8, 1]> = xavier_uniform();
}
```

Each `param` is a trainable tensor. Its canonical path, `TwoLayer.w1` and `TwoLayer.w2`, is the
name the IR and the training output use for it, and what its initial value is drawn from.
`xavier_uniform()` needs a rank-2 parameter; `zeros()` and `ones()` take a parameter of any rank
from 1, such as the bias `b: Tensor<f32, [H]>` of a dense layer, and draw nothing.

## 2. The forward function

Put `forward` inside the model, after the two `param` lines and before the model's closing `}`:

```modelspec
    fn forward(x: Tensor<f32, [16, 4]>)
        -> Tensor<f32, [16, 1]> {
        let h = relu(matmul(x, w1));
        return matmul(h, w2);
    }
```

Every type is written out, and every dimension is a number: `[16, 4]` times `[4, 8]` is
`[16, 8]`, `relu` keeps that shape, and `[16, 8]` times `[8, 1]` is `[16, 1]`, the declared return
type. Arithmetic is a call: `add(a, b)`, not `a + b`, which is refused with `E-SYNTAX-009`.

## 3. The train block

Add a `train` block after the model:

```modelspec
train FitTwoLayer {
    model = TwoLayer;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, [16, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 200;
    seed = 7;
    device = cpu;
}
```

The `input` has the type of `forward`'s argument and the `target` the type of its result. The loss
calls `model.forward` once and compares its result with the target. [The language](../concepts/language.md#the-train-block)
lists every setting and its rule.

## 4. Check it

```sh
modelspec check two_layer.mspec
```

It prints nothing and exits `0`. Now break it on purpose: change `w2` to `Tensor<f32, [4, 1]>`
and check again. The second `matmul` no longer fits, and `check` prints this on stderr and exits
`1`:

```text
E-TYPE-012: matmul contracting dimension mismatch
  two_layer.mspec:8:16
  lhs: Tensor<f32, [16, 8]>
  rhs: Tensor<f32, [4, 1]>
  expected rhs dimension[0] = 8; got 4
```

[Read a diagnostic](./read-a-diagnostic.md) explains each line. Change it back.

## 5. Look at the IR

```sh
modelspec ir two_layer.mspec --stage tensor
modelspec ir two_layer.mspec --stage training
```

The Tensor IR has one line per operation, with `forward` inlined: two `ParamRead`s, two `MatMul`s,
a `Relu` and the `Mse`. The Training IR adds the gradient of each weight and the two SGD updates.
[The pipeline](../concepts/pipeline.md) reads both.

## 6. Train it

```sh
modelspec train two_layer.mspec --backend cpu
```

It prints both initial weights, the loss at steps 0, 50, 100, 150 and 200, and both trained
weights. The loss falls from about 1.38 to about 0.018. The data are synthetic: the target is the
model's own prediction under a second, hidden set of weights drawn from the seed
([Training](../concepts/training.md)). Change `seed` and every number changes; keep it and every
run prints the same bytes.

`--backend vulkan` trains the same model on a Vulkan device and prints the same lines within the
GPU tolerance of the CPU run. It is M1, released since 0.4.0; a binary built without the `vulkan` feature refuses
it with `E-TARGET-001` ([Roadmap](../roadmap.md)).

## Where to go from here

- [Operators](../reference/operators.md): every operator and the shapes each accepts.
- [Diagnostics](../reference/diagnostics.md): every code the checker can print.
- [Syntax beyond M0](../examples/beyond-m0.md): modules, layers and generic dimensions, which M0
  refuses.
