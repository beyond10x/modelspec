---
title: Training
description: What modelspec train does in M0 - seeded initialization, synthetic teacher data, SGD on the CPU reference - and why two runs print the same bytes.
---

# Training

`modelspec train <file> --backend cpu` compiles the file to its [Training IR](./pipeline.md#training-ir)
and runs that step `steps` times on the CPU reference interpreter. The steps, learning rate and
seed come from the `train` block; the backend comes from `--backend`, not from the block's
`device`.

## What it prints

One line each, to stdout:

- `init @<path>: <type> = [...]` for every parameter, before the first step;
- `step <k> loss <v>` at step 0, every 50 steps and at the last step, where `k` is the number of
  updates made so far;
- `param @<path>: <type> = [...]` for every parameter, after the last step.

Parameters are listed by canonical path, values in row-major order, each `f32` in its shortest
decimal form that reads back to the same value.

## Where the numbers come from

Every random value is addressed, not drawn in sequence. DESIGN.md § 2.3 defines the derivation:
the counter-based generator Philox4x32-10, keyed by the train block's `seed`, with a counter made
of the element index, a stream tag and the 64-bit id of what is drawn. A parameter's id is the
first 8 bytes of the `blake3` hash of its canonical path. So `LinearReg.w` gets the same initial
value under `seed = 42` however the file is laid out and in whatever order the compiler meets it.

| Stream | What is drawn | How |
|---|---|---|
| `0` | the initial parameters | each parameter's initializer: `xavier_uniform` is uniform on `±sqrt(6 / (In + Out))` |
| `1` | the teacher parameters | the same initializer, on a different stream |
| `2` | the input | standard normal, by Box-Muller |

## Synthetic data

M0 has no data file format; that is an open question in DESIGN.md § 12. Instead, `train`
makes up a problem with a known answer. It draws the input, draws a second set of parameters, the
teacher, and computes the target as the model's prediction for that input under the teacher. The
teacher is the solution training should find.

For the linear regression example the trained weight ends within `1e-2` of the teacher weight, on
every element. The test `training_the_example_reaches_the_teacher` in
[`crates/modelspec-cli/tests/train_linear.rs`](https://github.com/beyond10x/modelspec/blob/main/crates/modelspec-cli/tests/train_linear.rs)
draws the teacher independently of the command line and checks it.

The train block must declare exactly one `input` and one `target`, the target must have the
model's output type, and the loss must read both. Each rule has its code, `E-DATA-001` to
`E-DATA-003` ([Diagnostics](../reference/diagnostics.md#training-data)).

## Reproducible to the byte

Two runs with the same file and seed print identical bytes. That holds because the arithmetic is
fixed as well as the draws: DESIGN.md § 2.2 fixes the reduction order of the CPU reference. A sum
adds its terms in row-major order, `matmul` accumulates over the contracting index in ascending
order, every intermediate is rounded to `f32`, and there is no fused multiply-add. The test
`two_runs_with_seed_42_print_identical_bytes` holds it.

A GPU backend will not be bit-identical to the CPU reference. DESIGN.md § 9 holds it to the CPU
results within documented tolerances instead, and that comparison is part of M1
([Roadmap](../roadmap.md)).

## Optimizer

M0 has one optimizer, `sgd(lr = <float>)`, with a positive learning rate that fits `f32`. The update
`w - lr * dL/dw` is part of the Training IR, not a step the runtime adds around it. AdamW is planned
for M3.
