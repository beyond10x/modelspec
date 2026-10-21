---
title: The pipeline
description: From a .mspec file to a verified training step, stage by stage, with the Tensor IR and Training IR of the linear regression example read line by line.
---

# The pipeline

The compiler turns a source file into a chain of immutable values, each verified before the next
stage reads it. DESIGN.md § 3 draws the whole chain; 0.1.0 builds it as far as the Training IR and
runs that on the CPU interpreter.

```text
.mspec source
  -> lexer, parser                 Source AST         built
  -> name resolution               names bound        built
  -> type, shape and train checks  Typed HIR          built
  -> lowering                      Tensor IR          built
  -> autodiff + optimizer step     Training IR        built
  -> CPU interpreter               a trained model    built
  -> fuse, schedule, tile          Kernel IR          M1, M2
  -> SPIR-V + Vulkan pipelines     GPU execution      M1
```

Both IR levels are verified when they are built, and dead code is removed from the Tensor IR.
`modelspec train --backend cpu` runs the CPU interpreter. Kernel IR and GPU execution are not
built.

`modelspec check` runs the first three stages. `modelspec ir --stage tensor` and `--stage
training` print the two IR levels, and the tests compare both byte for byte with their golden
files,
[`linear_regression.tensor.ir`](https://github.com/beyond10x/modelspec/blob/main/crates/modelspec-cli/tests/golden/linear_regression.tensor.ir)
and
[`linear_regression.training.ir`](https://github.com/beyond10x/modelspec/blob/main/crates/modelspec-cli/tests/golden/linear_regression.training.ir).

## Tensor IR

Tensor IR is typed SSA: every value has one definition, a type and a shape, and refers to earlier
values by number. For [the linear regression example](../examples/linear-regression.md),
`modelspec ir examples/linear_regression.mspec --stage tensor` prints:

```text
param @LinearReg.w: Tensor<f32, [4, 1]> = xavier_uniform
%0: Tensor<f32, [8, 4]> = Input "x"
%1: Tensor<f32, [8, 1]> = Input "y"
%2: Tensor<f32, [4, 1]> = ParamRead @LinearReg.w
%3: Tensor<f32, [8, 1]> = MatMul %0, %2
%4: Tensor<f32, []> = Mse %3, %1
output %4
```

| Line | What it is |
|---|---|
| `param @LinearReg.w` | the parameter, by canonical path, with its initializer |
| `%0`, `%1` | the train block's `input` and `target` |
| `%2` | a read of the parameter's current value |
| `%3` | `model.forward(x)`, inlined: the function call is gone, only its operation is left |
| `%4` | the loss; `output` marks it as the program's result |

Each op is typed by the same function the checker uses, `ops::infer` in the operator registry, so
the IR cannot hold a shape the checker would have refused. The verifier checks every value's type
against its operands, and dead-code elimination drops anything the loss does not reach.

## Training IR

The Training IR is the whole training step as one program: forward pass, loss, gradients and the
optimizer update. `--stage training` prints:

```text
param @LinearReg.w: Tensor<f32, [4, 1]> = xavier_uniform {v0}
optimizer sgd lr = 0.1
forward %0: Tensor<f32, [8, 4]> = Input "x"
forward %1: Tensor<f32, [8, 1]> = Input "y"
forward %2: Tensor<f32, [4, 1]> = ParamRead @LinearReg.w {!0 read @LinearReg.w v0}
forward %3: Tensor<f32, [8, 1]> = MatMul %0, %2
loss %4: Tensor<f32, []> = Mse %3, %1
loss %5: Tensor<f32, []> = Scalar 1.0
forward %6: Tensor<f32, [8, 1]> = Sub %3, %1
forward %7: Tensor<f32, []> = Scalar 0.25
gradient %8: Tensor<f32, []> = Mul %5, %7
gradient %9: Tensor<f32, [8, 1]> = Mul %6, %8
forward %10: Tensor<f32, [4, 8]> = Transpose %0
gradient %11: Tensor<f32, [4, 1]> = MatMul %10, %9
optimizer %12: Tensor<f32, []> = Scalar 0.1
optimizer %13: Tensor<f32, [4, 1]> = Mul %12, %11
optimizer %14: Tensor<f32, [4, 1]> = Sub %2, %13 {!1 write @LinearReg.w v1}
output %4 loss
output %11 gradient @LinearReg.w
output %14 update @LinearReg.w
```

Read from the loss backwards:

- `%5` is the seed of reverse mode, `dL/dL = 1`.
- `%6` to `%9` are the gradient of `mse` with respect to the prediction, `(A - B) * (G * (2 / n))`:
  `%7` is `2 / 8`, because the prediction has 8 elements.
- `%10` and `%11` are the gradient of `matmul` with respect to its right operand,
  `transpose(A) @ G`. `%11` is `dL/dw`.
- `%12` to `%14` are the SGD update, `w - lr * dL/dw`.

The first word of each line is its subgraph: `forward`, `loss`, `gradient` or `optimizer`. The
subgraphs are there for analysis and verification. They are not execution boundaries, which is what
lets a later milestone fuse across them.

The parameter has versions. The forward read takes `v0` (`{!0 read @LinearReg.w v0}`) and the
update writes `v1` (`{!1 write @LinearReg.w v1}`). These numbered effects order every read and
write of state, so no rewrite can overwrite a value that the backward pass still needs. The
gradient rules themselves are one table in the operator registry
([Operators § Gradients](../reference/operators.md#gradients)).

## What comes after M0

Kernel IR, the execution DAG, SPIR-V generation and Vulkan pipeline creation are the M1 and M2
stages of DESIGN.md § 3.1 and are not built. The GPU backend will consume the same verified
Training IR the CPU interpreter runs today, which is the first invariant of the design: one
semantic core for every backend ([Roadmap](../roadmap.md)).
