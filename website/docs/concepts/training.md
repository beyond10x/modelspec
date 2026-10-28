---
title: Training
description: What modelspec train does - seeded initialization, synthetic, dense or packed data, SGD on the CPU reference or a Vulkan GPU - and why two runs print the same bytes.
sidebar_position: 4
lede: "modelspec train runs the compiled training step on the CPU reference or one Vulkan device, and every random value is addressed by seed, stream and id, so two CPU runs with the same file, data and seed print identical bytes."
source: "DESIGN.md § 2.2, § 2.3, § 9 and ADR-009; crates/modelspec-cli/src/train.rs and crates/modelspec-runtime/src/train and cpu; the tests train_linear.rs, train_dense_data.rs, train_data.rs and train_dropout.rs of crates/modelspec-cli."
---

# Training

`modelspec train <file> --backend cpu` compiles the file to its [Training IR](./pipeline.md#training-ir)
and runs that step `steps` times on the CPU reference interpreter; `--backend vulkan` runs it on
one Vulkan device. The steps, learning rate and seed come from the `train` block; the backend
comes from `--backend`, not from the block's `device`.

## What it prints

One line each, to stdout:

- `data dataset <id> tokenizer <id> seq_len <S> masking ... binding <id>`, on a packed `--data`
  directory only, before anything else;
- `init @<path>: <type> = [...]` for every parameter, before the first step;
- `step <k> loss <v>` at step 0, every 50 steps and at the last step, where `k` is the number of
  updates made so far;
- `param @<path>: <type> = [...]` for every parameter, after the last step;
- `eval loss=<mean loss> accuracy=<fraction>` after them, with `--eval`.

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
| `0` | the initial parameters | each parameter's initializer: `xavier_uniform` is uniform on `±sqrt(6 / (In + Out))`; `zeros` and `ones` draw nothing |
| `1` | the teacher parameters of synthetic data | the same initializer, on a different stream |
| `2` | the input of synthetic data | standard normal, by Box-Muller |
| `3` | the masking of a packed batch | per step: which tokens are selected, and whether each is masked, replaced or kept |
| `4` | the dropout masks | per call site and step |
| `5` | the order of the packed `Train` sequences | per epoch |

## Data

`train` reads one of three sources.

| Source | How | The `train` block |
|---|---|---|
| synthetic | the default | exactly one `input` and one `target` |
| dense files | `--data <dir>` with `<name>.f32` per `f32` declaration and `<name>.i32` per `i32` declaration, row-major little-endian, and an optional `order.u32` of sample indices | any number of `input` and `target` declarations, at least one |
| packed sequences | `--data <dir>` on a directory `modelspec data pack` wrote, recognised by its `binding.json` | each declaration names a tensor of the masked-language-model batch: `ids`, `positions`, `key_mask`, `labels` or `weights` |

Synthetic data is a problem with a known answer. `train` draws the input, draws a second set of
parameters, the teacher, and computes the target as the model's prediction for that input under
the teacher. The teacher is the solution training should find. For the linear regression example
the trained weight ends within `1e-2` of the teacher weight, on every element. The test
`training_the_example_reaches_the_teacher` in
[`crates/modelspec-cli/tests/train_linear.rs`](https://github.com/beyond10x/modelspec/blob/main/crates/modelspec-cli/tests/train_linear.rs)
draws the teacher independently of the command line and checks it.

On a packed directory, step `k` feeds the `Train` batch of step `k`: the sequences in the epoch
order of stream 5, 15% of their tokens selected on stream 3, and of those 80% replaced by the mask
token, 10% by a random token and 10% left as they are. [Prepare training data](../guides/prepare-training-data.md)
builds a packed directory and trains on it.

In a dense directory, a sample is one row along the leading dimension every declaration shares;
when they share none, a sample is the whole declared tensor. Without `order.u32`, step `k` reads
block `k` of every file, wrapping at the end. Before the first step, every value of an `i32`
file is checked against the rows of each `embedding` table and the classes of each
`cross_entropy` that read it, so an id or label out of range is refused before anything trains.

`--init <dir>` starts each parameter that has a file `<dir>/<P>.f32`, `P` its path without the model name, from that file, and
`--eval <dir>` runs the forward program over every sample of a dense directory after the last
step and prints the mean batch loss, with an accuracy only when the loss compares the model
output with the block's one `f32` target; a classifier whose one target is `i32` labels prints
the loss alone. `--init` runs on `--backend cpu` only, `--data` and `--eval`
on both backends; `--eval` beside a packed `--data` is a usage error. Each refusal of a data source has its code, `E-DATA-001` to
`E-DATA-009` and `E-DATA-011` ([Diagnostics](../reference/diagnostics.md#training-data)).

## Reproducible to the byte

Two runs with the same file, data and seed print identical bytes. That holds because the
arithmetic is fixed as well as the draws: DESIGN.md § 2.2 fixes the reduction order of the CPU
reference. A sum adds its terms in row-major order, `matmul` accumulates over the contracting
index in ascending order from its first product, each later product fused into the sum with one
IEEE fused multiply-add (DESIGN.md § 12 ADR-009), and every other intermediate is rounded to
`f32`. The test `two_runs_with_seed_42_print_identical_bytes` holds it.

The Vulkan backend is not bit-identical to the CPU reference. DESIGN.md § 9 holds it to the CPU
results within documented tolerances instead, and the tests of `modelspec-runtime` compare every
loss and final parameter of a device run with the CPU run.

## Dropout

`dropout(x, p = 0.1)` applies in the training step only. Its mask comes from stream 4, keyed by
the seed, the call site and the step, and its gradient recomputes the same mask instead of storing
it. The final loss line, `--eval` and `modelspec infer` run the forward program without it, and
`p = 0.0` lowers to the identity.

## Optimizer

There are three optimizers, and each update is part of the Training IR, not a step the runtime
adds around it.

- `sgd(lr = <float>)`, with a positive learning rate that fits `f32`, updates `w - lr * dL/dw`
  at a constant rate.
- Since 0.12.0, `adamw(lr = F, weight_decay = F, beta1 = F, beta2 = F, eps = F, warmup_steps = N)`
  trains with AdamW and decoupled weight decay on the CPU reference, compiled into the training
  step's `Optimizer` subgraph. Each parameter's first and second moments start at 0 and are
  versioned with it, and the learning rate rises linearly from 0 over `warmup_steps` and falls
  linearly to 0 at the run's last step. `E-TYPE-023` and `E-TYPE-024` refuse out-of-range
  arguments, `E-TRAIN-001` also stops the run when a moment is not finite, and `train
  --checkpoint` and `--resume` refuse an AdamW run because a checkpoint does not carry the moments
  yet.
- `muon(lr = F, momentum = F, weight_decay = F, ns_steps = N, warmup_steps = N, adam_lr = F,
  beta1 = F, beta2 = F, eps = F)` trains each parameter of rank 2 that no `embedding` reads with
  Muon on the CPU reference, and every other parameter, a tied embedding table included, with
  AdamW at `adam_lr`. A Muon parameter keeps a momentum buffer `b`, starting at 0 and versioned
  with it; each step takes the Nesterov momentum `u = g + momentum * b`, divides it by its
  Frobenius norm, orthogonalizes it with `ns_steps` Newton–Schulz steps and applies it scaled by
  `0.2 * sqrt(max(rows, cols))`, with the decoupled weight decay. Both rates follow the warm-up
  schedule of `adamw`. `ir --stage training` prints which update each parameter takes.
  `E-TYPE-025` refuses a `momentum` outside `[0, 1)` and an `ns_steps` outside `[1, 10]`,
  `E-TRAIN-001` also stops the run when a momentum buffer or the Frobenius norm of a step is not
  finite, and `train --checkpoint` and `--resume` refuse a Muon run.

:::caution[Planned]

Muon on a Vulkan device is planned; `--backend vulkan` answers `E-TARGET-011` for it. AdamW trains
on the device.

:::
