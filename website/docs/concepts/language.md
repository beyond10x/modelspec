---
title: The language
description: A tour of the .mspec language from DESIGN.md § 2, and the M0 subset the 0.1.0 compiler accepts.
---

# The language

A `.mspec` file is declarative, statically typed and pure by default. It says what a model
computes and how it is trained; it does no I/O and has no runtime objects. This page is a tour of
[DESIGN.md § 2](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#2-modelspec-source-language),
in two parts: the M0 subset that 0.1.0 parses and checks, and the target language it grows into.

## An M0 file

An M0 file holds one `model` and one `train` block, in either order.

```modelspec
model TwoLayer {
    param w1: Tensor<f32, [4, 8]> = xavier_uniform();
    param w2: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 4]>)
        -> Tensor<f32, [16, 1]> {
        let h = relu(matmul(x, w1));
        return matmul(h, w2);
    }
}

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

### The model

| Construct | Meaning in M0 |
|---|---|
| `param name: Tensor<...> = xavier_uniform();` | a trainable tensor with a stable canonical path, here `TwoLayer.w1`. `xavier_uniform()` needs a rank-2 parameter; `zeros()` and `ones()` take any rank from 1 and draw nothing |
| `fn name(arg: Tensor<...>, ...) -> Tensor<...> { ... }` | a pure, typed function: any number of `let` bindings, then one `return`. A model may have several; the train block calls `forward` |
| `let h = expr;` | names an intermediate value. There is no assignment and no mutation |
| `matmul(a, b)`, `relu(a)`, ... | an operator call with positional arguments. M0 has nine operators and no infix syntax, so `a + b` is written `add(a, b)` ([Operators](../reference/operators.md)) |
| `0.5` | a float literal, typed as a scalar `f32` tensor, for example `mul(x, 0.5)`. An integer literal where a tensor is expected is refused (`E-TYPE-005`): write `1.0`, not `1` |

### Types and shapes

Every value has the type `Tensor<f32, [d1, d2, ...]>`. Every dimension is a positive integer
literal, and `[]` is a scalar. `f32` is the only element type. The shape rules are strict on
purpose (DESIGN.md § 2.2):

- `matmul([M, K], [K, N])` is `[M, N]`, for rank-2 operands only.
- `add`, `sub` and `mul` take two operands of the same shape, or a scalar and a tensor. There is no
  other broadcasting.
- `relu` keeps its operand's shape; `transpose([M, N])` is `[N, M]`.
- `sum`, `mean` and `mse` return a scalar; `mse(a, b)` needs `a` and `b` of the same shape.

A call that breaks a rule is a compile error with its own code, the source position and both
operand types ([a shape error](../examples/shape-error.md)).

### The train block

| Setting | Meaning in M0 |
|---|---|
| `model = Name;` | the model to train |
| `input x: Tensor<...>;`, `target y: Tensor<...>;` | exactly one of each. The target has the model's output type, and the loss reads both |
| `loss = expr;` | a scalar expression that calls `model.forward` exactly once (`E-TYPE-017` otherwise) |
| `optimizer = sgd(lr = 0.05);` | plain SGD with a positive learning rate; the only optimizer |
| `steps = 200;` | the number of updates, at least one |
| `seed = 7;` | the seed of every random draw: initial parameters and synthetic data ([Training](./training.md)) |
| `device = cpu;` | `cpu` or `vulkan(<index>)`. The 0.1.0 command line takes the backend from `--backend` instead, and refuses `vulkan` |

## The target language

DESIGN.md § 2 designs a larger language. This model is its running example, and 0.1.0 refuses it:

```modelspec
module Linear<In: dim, Out: dim> {
    param weight: Tensor<f32, [In, Out]> = xavier_uniform();

    fn forward<B: dim>(x: Tensor<f32, [B, In]>)
        -> Tensor<f32, [B, Out]> {
        return matmul(x, weight);
    }
}

model TinyMLP<B: dim> {
    layer fc1: Linear<32, 64>;
    layer fc2: Linear<64, 1>;

    fn forward(x: Tensor<f32, [B, 32]>)
        -> Tensor<f32, [B, 1]> {
        let h = relu(fc1.forward(x));
        return fc2.forward(h);
    }
}
```

| Construct | Meaning | M0 |
|---|---|---|
| `module` | a reusable group of parameters and functions | refused, `E-SYNTAX-004` |
| `layer` | a named module instance, elaborated statically; each instance owns its own weights | refused, `E-SYNTAX-005` |
| `<B: dim>`, `[B, 32]` | a generic dimension, resolved at compile time | refused, `E-SYNTAX-006` to `E-SYNTAX-008` |
| `const` | an immutable scalar or tensor | refused, `E-SYNTAX-010` |
| `infer` | an inference signature | refused, `E-SYNTAX-002` (`expected model or train`) |
| `target` (top level) | backend constraints kept out of model semantics | later milestones; `target y:` inside `train` is M0 |

The parser does not answer these with a generic syntax error. Each construct beyond M0 has its own
code naming what is missing, and [Syntax beyond M0](../examples/beyond-m0.md) shows the output. The
[diagnostic reference](../reference/diagnostics.md) lists every code.

Two rules hold for the whole language. Composition does not decide kernel boundaries: functions
and layers are inlined, and the compiler may fuse or split the operations they contain. And every
parameter keeps a stable canonical path such as `TinyMLP.fc1.weight`, and its initial value is
drawn from that path and the seed, not from the order in which the compiler meets it.
