---
title: BERT operators
description: The operators a small BERT needs - embedding, add_rows, gelu, layer_norm, softmax, attention, cross_entropy and dropout - what each computes, its i32 operands, its gradient, and why it runs on the CPU only for now.
---

# BERT operators

ModelSpec's current milestone is a small BERT, pre-trained from scratch. Eight operators were
added for it, on top of the nine that a multilayer perceptron needs. Each is typed by the operator
registry, has a reverse-mode gradient checked against central finite differences in `f64` and
`f32`, and runs on the CPU interpreter. [Operators](../reference/operators.md) has the shape rule,
accepted and refused examples and the gradient formula of each; this page says what they are for.

| Operator | Computes | Released |
|---|---|---|
| `embedding(table, ids)` | row `ids[n]` of `table: f32[V, H]` for each of `ids: i32[N]`, so `f32[N, H]`. The token and position tables of BERT | 0.10.0 |
| `add_rows(x, b)` | `b: f32[H]` added to every row of `x: f32[N, H]`: the bias of a dense layer | 0.10.0 |
| `gelu(x)` | the tanh form of GELU, `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, element by element | 0.10.0 |
| `layer_norm(x, gamma, beta)` | each row of `x: f32[N, H]` normalized to mean 0 and variance 1, `eps = 1e-12`, then scaled by `gamma` and shifted by `beta` | 0.10.0 |
| `softmax(x)` | the softmax of each row of a rank-2 `x`, the row maximum subtracted before `exp` | 0.10.0 |
| `attention(q, k, v, key_mask, heads = A)` | multi-head scaled dot-product attention over sequences of `S` rows: `q`, `k`, `v: f32[N, H]`, `key_mask: f32[B, S]` with `N = B · S`, `A` heads of width `H / A` | 0.10.0 |
| `cross_entropy(logits, labels, weights)` | the weighted masked-language-model loss over `logits: f32[N, V]`, `labels: i32[N]` and `weights: f32[N]`, a scalar | not yet released |
| `dropout(x, p = 0.1)` | each element zeroed with probability `p` and the rest scaled by `1 / (1 - p)`, in the training step only | not yet released |

## Row ids and labels are `i32`

`embedding` reads `i32` ids and `cross_entropy` reads `i32` labels; every other operand is `f32`.
An `i32` tensor is declared by an `input` or `target` or a function parameter and used only there
(`E-TYPE-018` otherwise), and it takes no gradient. An id outside the table or a label outside the
classes is a runtime error that names the row, before the operator runs.

The operands that take no gradient must not depend on a parameter: a `key_mask` of `attention` or
the `labels` or `weights` of `cross_entropy` whose value reads a parameter, through any chain of
`let`s and function calls, is `E-TYPE-021`.

## Masks and weights

The key mask of `attention` holds 1 for a key a query may attend to and 0 for one it may not, per
sequence; any other value, or a sequence with every key masked, is a runtime error rather than a
NaN. The weights of `cross_entropy` select the rows that count: on a packed batch they are 1 at a
selected token and 0 elsewhere, so the loss is the mean over the selected tokens. A weight sum of
0 gives the loss 0 and a zero gradient.

## Deterministic sums

Every row reduction of these operators sums in ascending order from the first element, as every
reduction of the CPU reference does ([Training](./training.md#reproducible-to-the-byte)): the mean
and variance of `layer_norm`, the row sums of `softmax` and of the bias gradient `SumRows`, and
the rows of the `embedding` gradient, added in ascending `n`. Two runs with one seed print the same
bytes.

## A model that uses them

The test model `mlm_bag.mspec` is the smallest masked-language model: a token table, tied as its
own decoder, with no layers between.

```modelspec title="crates/modelspec-cli/tests/fixtures/mlm_bag.mspec"
model MlmBag {
    param table: Tensor<f32, [300, 16]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [64]>) -> Tensor<f32, [64, 300]> {
        return matmul(embedding(table, ids), transpose(table));
    }
}

train Fit {
    model = MlmBag;
    input ids: Tensor<i32, [64]>;
    target labels: Tensor<i32, [64]>;
    target weights: Tensor<f32, [64]>;
    loss = cross_entropy(model.forward(ids), labels, weights);
    optimizer = sgd(lr = 1.0);
    steps = 200;
    seed = 7;
    device = cpu;
}
```

It trains on a packed directory; [Prepare training data](../guides/prepare-training-data.md)
builds one and runs it.

## No Vulkan kernel yet

None of the eight has a Vulkan kernel. A program that uses one checks, compiles for the CPU and
trains with `--backend cpu`; every command that needs a kernel stops at the first such call with
`E-TARGET-011`, before a device is opened:

```sh
modelspec ir mlm_bag.mspec --stage kernel
```

```text
E-TARGET-011: operator `embedding` has no Vulkan kernel yet
  mlm_bag.mspec:5:23
  compile and train it with `--backend cpu`
```

It exits `1`. `compile --backend vulkan`, `train --backend vulkan` and `infer --backend vulkan`
answer the same.

:::caution[Planned]

Vulkan kernels for `embedding`, `add_rows`, `gelu`, `layer_norm`, `softmax` and `dropout`, checked
against the CPU interpreter, then kernels for `attention` and `cross_entropy` and a full BERT step
on the GPU. None is built ([Roadmap](../roadmap.md)).

:::
