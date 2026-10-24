---
title: Getting started
description: Download the modelspec 0.9.0 binary, verify it, and check and train the example model on the CPU reference or a Vulkan GPU.
---

# Getting started

## Install the 0.9.0 binary

The [0.9.0 release](https://github.com/beyond10x/modelspec/releases/tag/0.9.0) carries one
prebuilt binary, for Linux on x86_64, and its SHA-256 checksum. There is no macOS, Windows or
aarch64 binary; on those, [build from source](#build-from-source).

```sh
base=https://github.com/beyond10x/modelspec/releases/download/0.9.0
bin=modelspec-0.9.0-x86_64-unknown-linux-gnu
curl -fsSLO "$base/$bin"
curl -fsSLO "$base/$bin.sha256"
sha256sum -c "$bin.sha256"
install -D -m 755 "$bin" ~/.local/bin/modelspec
modelspec --version
```

`sha256sum -c` prints `modelspec-0.9.0-x86_64-unknown-linux-gnu: OK`, and `modelspec --version`
prints `modelspec 0.9.0`. `install -D` creates `~/.local/bin` if it is missing; that directory
must be on your `PATH` for the last line to find the binary. The binary links only the C runtime
(`libc`, `libm` and `libgcc_s`), which every glibc Linux system has. `train --backend vulkan` and
`inspect-device` open the system Vulkan loader when they run; a machine without one gets
`E-TARGET-002`, and everything else works without a GPU driver.

## Check and train the example

Save the example of the repository,
[`examples/linear_regression.mspec`](https://github.com/beyond10x/modelspec/blob/main/examples/linear_regression.mspec),
as `linear_regression.mspec`, then:

```sh
modelspec check linear_regression.mspec
modelspec train linear_regression.mspec --backend cpu
```

`check` prints nothing and exits `0`. `train` draws the weight and a batch of synthetic data from
`seed = 42`, prints the loss at step 0, every 50 steps and at step 500, and ends with the trained
weight:

```text
init @LinearReg.w: Tensor<f32, [4, 1]> = [0.9730758, -0.34559327, -0.86724, 0.14564086]
step 0 loss 9.693205
step 50 loss 0.0032300856
...
step 500 loss 0.0000000000561049
param @LinearReg.w: Tensor<f32, [4, 1]> = [-0.9486646, -0.25639856, 0.4230444, 1.059935]
```

The same seed prints the same bytes on every run. [Linear regression](./examples/linear-regression.md)
shows the full output, and the IR between the two commands.

## Train on a GPU

With a Vulkan 1.2 driver installed, save
[`examples/mnist_mlp.mspec`](https://github.com/beyond10x/modelspec/blob/main/examples/mnist_mlp.mspec)
and train it on the first Vulkan device:

```sh
modelspec inspect-device --backend vulkan
modelspec train mnist_mlp.mspec --backend vulkan --stats
```

`train` prints the same lines as `--backend cpu`, within the GPU tolerance of the CPU reference,
and `--stats` adds the run counters on stderr. [Inspect a run](./guides/inspect-a-run.md) shows
the kernel plan and the per-dispatch timing of `--profile`.

## Look at the IR

```sh
modelspec ir linear_regression.mspec --stage tensor
modelspec ir linear_regression.mspec --stage training
```

The first prints the forward program as typed SSA; the second adds the gradient and the SGD
update. [The pipeline](./concepts/pipeline.md) reads both line by line.

## Build from source

You need a Rust toolchain that supports edition 2024 (Rust 1.85 or later) and nothing else: no
Vulkan SDK, no Python. `modelspec train --backend vulkan` and `inspect-device` open the system
Vulkan loader when they run; a machine without one gets `E-TARGET-002`.

```sh
git clone https://github.com/beyond10x/modelspec
cd modelspec
cargo build --release -p modelspec-cli
./target/release/modelspec check examples/linear_regression.mspec
```

The repository's own gate is `task ci` ([go-task](https://taskfile.dev)), which builds, lints and
tests the workspace and checks that this site's generated pages match the code.

## Exit status

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the file has diagnostics, or training failed |
| `2` | a usage error: an unknown command, option or value |

## Next

- [Write a model](./guides/write-a-model.md): a model of your own in the M0 subset.
- [Read a diagnostic](./guides/read-a-diagnostic.md): what a refusal tells you.
- [CLI reference](./reference/cli.md): every command and option.
