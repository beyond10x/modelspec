---
title: The language
description: A tour of the .mspec language from DESIGN.md § 2, the subset the 0.16.0 compiler accepts, and the target language it grows into.
sidebar_position: 1
lede: "A .mspec file is declarative, statically typed and pure, and 0.16.0 parses and checks a subset of the language DESIGN.md § 2 designs, refusing each construct beyond it with its own code."
source: "DESIGN.md § 2; crates/modelspec-core/src/syntax, resolve.rs, types.rs and ops.rs; the tests syntax.rs of crates/modelspec-core and check.rs, check_i32.rs, check_attention.rs and check_dropout.rs of crates/modelspec-cli; the diagnostic reference generated from crates/modelspec-core/src/diag.rs."
---

# The language

A `.mspec` file is declarative, statically typed and pure by default. It says what a model
computes and how it is trained; it does no I/O and has no runtime objects. This page is a tour of
[DESIGN.md § 2](https://github.com/beyond10x/modelspec/blob/main/DESIGN.md#2-modelspec-source-language),
in two parts: the subset that 0.16.0 parses and checks, and the target language it grows into.
The subset is called M0, after the milestone that defined its syntax; later releases added
operators, the `i32` element type and named operator arguments to it, and no new statements.

## A file of the subset

A file holds one `model` and one `train` block, in either order.

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

| Construct | Meaning |
|---|---|
| `param name: Tensor<...> = xavier_uniform();` | a trainable tensor with a stable canonical path, here `TwoLayer.w1`. `xavier_uniform()` needs a rank-2 parameter; `zeros()` and `ones()` take any rank from 1 and draw nothing. A parameter is always `f32` |
| `fn name(arg: Tensor<...>, ...) -> Tensor<...> { ... }` | a pure, typed function: any number of `let` bindings, then one `return`. A model may have several; the train block calls `forward` |
| `let h = expr;` | names an intermediate value. There is no assignment and no mutation |
| `matmul(a, b)`, `attention(q, k, v, mask, heads = 2)` | an operator call: positional operands, then the operator's named compile-time arguments, each a literal given once. There is no infix syntax, so `a + b` is written `add(a, b)`. [Operators](../reference/operators.md) lists every operator and its rule |
| `0.5` | a float literal, typed as a scalar `f32` tensor, for example `mul(x, 0.5)`. An integer literal where a tensor is expected is refused (`E-TYPE-005`): write `1.0`, not `1` |

### Types and shapes

Every value has the type `Tensor<f32, [d1, d2, ...]>` or `Tensor<i32, [d1, ...]>`. Every
dimension is a positive integer literal, and `[]` is a scalar. An `i32` tensor holds row ids or
labels: it is declared only by an `input` or `target` of the train block or by a function
parameter, and read only as the ids of `embedding` or the labels of `cross_entropy`. Any other use
of an `i32`, and an `f32` where ids or labels go, is `E-TYPE-018`. The shape rules are strict on
purpose (DESIGN.md § 2.2):

- `matmul([M, K], [K, N])` is `[M, N]`, for rank-2 operands only.
- `add`, `sub` and `mul` take two operands of the same shape, or a scalar and a tensor. There is no
  other broadcasting.
- `relu`, `gelu` and `dropout` keep their operand's shape; `transpose([M, N])` is `[N, M]`.
- `sum`, `mean` and `mse` return a scalar; `mse(a, b)` needs `a` and `b` of the same shape.
- `embedding(table [V, H], ids i32[N])` is `[N, H]`.
- `add_rows([N, H], [H])` adds the row to every row; `layer_norm([N, H], gamma [H], beta [H])` is
  `[N, H]`; `softmax` takes a rank-2 operand and keeps its shape.
- `attention(q, k, v, key_mask, heads = A)` takes `q`, `k` and `v` of one shape `[N, H]` and a key
  mask `[B, S]` with `N = B · S`, and returns `[N, H]`; `A` must divide `H` (`E-TYPE-020`).
- `cross_entropy(logits [N, V], labels i32[N], weights [N])` returns a scalar.

A call that breaks a rule is a compile error with its own code, the source position and both
operand types ([a shape error](../examples/shape-error.md)). The BERT operators have their own
page, [BERT operators](./bert-operators.md).

### Named operator arguments

`attention` takes `heads`, an integer literal of at least 1, and `dropout` takes `p`, a float
literal in `[0, 1)`. Each is required, given once and after the operands. A value that is not a
literal of its kind is `E-TYPE-019`, a name the operator does not take or one given twice is
`E-TYPE-002`, and a `p` outside `[0, 1)` is `E-TYPE-022`.

### The train block

| Setting | Meaning |
|---|---|
| `model = Name;` | the model to train |
| `input x: Tensor<...>;`, `target y: Tensor<...>;` | on synthetic data and on dense `--data` files, exactly one of each: the target has the model's output type, and the loss reads both. On a packed `--data` directory, each declaration names a tensor of the masked-language-model batch, such as `ids`, `labels` and `weights` ([Prepare training data](../guides/prepare-training-data.md)) |
| `loss = expr;` | a scalar expression that calls `model.forward` exactly once (`E-TYPE-017` otherwise) |
| `optimizer = sgd(lr = 0.05);` | plain SGD with a positive learning rate, constant over the run |
| `optimizer = adamw(lr = F, weight_decay = F, beta1 = F, beta2 = F, eps = F, warmup_steps = N);` | AdamW with decoupled weight decay, every argument required and named; the learning rate rises linearly from 0 over `warmup_steps` and falls linearly to 0 at the last step. Out-of-range arguments are `E-TYPE-023` and `E-TYPE-024`. CPU reference only; `train --checkpoint` and `--resume` refuse an AdamW run |
| `optimizer = muon(lr = F, momentum = F, weight_decay = F, ns_steps = N, warmup_steps = N, adam_lr = F, beta1 = F, beta2 = F, eps = F);` | Muon: orthogonalized momentum (Nesterov momentum, the update normalised by its Frobenius norm, `ns_steps` Newton-Schulz steps, the scale `0.2 * sqrt(max(rows, cols))`, decoupled weight decay) for every rank-2 parameter that no `embedding` reads, AdamW at `adam_lr` for the rest; both rates on the `adamw` warm-up schedule. `E-TYPE-025` refuses a `momentum` outside [0, 1) or an `ns_steps` outside [1, 10]. CPU reference only; `train --checkpoint` and `--resume` refuse a Muon run |
| `steps = 200;` | the number of updates, at least one |
| `seed = 7;` | the seed of every random draw: initial parameters, synthetic data, the batches drawn from a packed directory and dropout masks ([Training](./training.md)) |
| `device = cpu;` | `cpu` or `vulkan(<index>)`. The command line takes the backend from `--backend`, not from this setting |

## The target language

DESIGN.md § 2 designs a larger language. This model is its running example, and 0.16.0 refuses
it:

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

| Construct | Meaning | 0.16.0 |
|---|---|---|
| `module` | a reusable group of parameters and functions | refused, `E-SYNTAX-004` |
| `layer` | a named module instance, elaborated statically; each instance owns its own weights | refused, `E-SYNTAX-005` |
| `<B: dim>`, `[B, 32]` | a generic dimension, resolved at compile time | refused, `E-SYNTAX-006` to `E-SYNTAX-008` |
| `const` | an immutable scalar or tensor | refused, `E-SYNTAX-010` |
| `infer` | an inference signature | refused, `E-SYNTAX-002` (`expected model or train`) |
| `target` (top level) | backend constraints kept out of model semantics | later milestones; `target y:` inside `train` is accepted |

The parser does not answer these with a generic syntax error. Each construct beyond the subset has
its own code naming what is missing, and [Syntax beyond M0](../examples/beyond-m0.md) shows the
output. The [diagnostic reference](../reference/diagnostics.md) lists every code.

Two rules hold for the whole language. Composition does not decide kernel boundaries: functions
and layers are inlined, and the compiler may fuse or split the operations they contain. And every
parameter keeps a stable canonical path such as `TinyMLP.fc1.weight`, and its initial value is
drawn from that path and the seed, not from the order in which the compiler meets it.
