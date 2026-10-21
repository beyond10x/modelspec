---
title: Read a diagnostic
description: The layout of every diagnostic modelspec prints, line by line, and where to look up a code.
---

# Read a diagnostic

`modelspec check` and `modelspec train` print the same diagnostics, to stderr, and exit `1` when
there is at least one. A clean file prints nothing. This is the diagnostic of
[a shape error](../examples/shape-error.md):

```text
E-TYPE-012: matmul contracting dimension mismatch
  crates/modelspec-docs/models/matmul_mismatch.mspec:6:16
  lhs: Tensor<f32, [8, 4]>
  rhs: Tensor<f32, [3, 1]>
  expected rhs dimension[0] = 4; got 3
```

| Line | What it says |
|---|---|
| `E-TYPE-012: matmul contracting dimension mismatch` | the code, then the message. The code is what to search for: the [reference](../reference/diagnostics.md#e-type-012) lists it with its meaning |
| `crates/.../matmul_mismatch.mspec:6:16` | the file, line and column of the construct, both counted from 1. Here it is the `matmul(x, w)` call |
| `lhs: ...`, `rhs: ...` | labelled details: the values the rule looked at, here the types of both operands |
| `expected rhs dimension[0] = 4; got 3` | an unlabelled detail: what the rule needed and what it found |

Every diagnostic has this layout (DESIGN.md § 8.1): one line with the code and message, then the
position when there is one, then one indented line per detail. A diagnostic that is not about a
place in the file has no position. `E-TARGET-001` is one, because `--backend vulkan` is refused
before the file is read:

```text
E-TARGET-001: the vulkan backend is not available in M0
  backend: vulkan
  available: cpu
```

## Several at once

The checker reports every diagnostic in the file, in source order, not only the first. An
expression that fails is reported once and not typed further, so one mistake does not cascade
into the expressions that contain it. The parser skips to the end of an item after its first
error in that item and goes on with the next. [Syntax beyond M0](../examples/beyond-m0.md) prints
three diagnostics for one file.

## What the prefix tells you

| Prefix | Stage | Usually means |
|---|---|---|
| `E-SYNTAX` | parser | a typo, or a construct the M0 subset does not have yet. Constructs of the target language such as `module`, `layer` and infix operators each have their own code |
| `E-NAME` | name resolution | an unknown variable, operator or function, or a name declared twice |
| `E-TYPE` | type and shape checker | operands whose shapes do not fit, a wrong argument count, or a train-block rule such as the loss calling `model.forward` more than once |
| `E-DATA` | `train`, after the check | the input and target do not fit the synthetic data loader |
| `E-TARGET` | `train`, before the check | the backend is not available |
| `E-INTERNAL` | the compiler | a compiler bug, not a mistake in the model; please [report it](https://github.com/beyond10x/modelspec/issues) with the file |

The [diagnostic reference](../reference/diagnostics.md) lists every code with its meaning and the
source file that declares it. It is generated from those declarations, so a code that exists is on
it.
