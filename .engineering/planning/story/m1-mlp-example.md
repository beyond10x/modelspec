---
format: aep.planning-md/3
id: story:m1-mlp-example
kind: story
status: implemented
title: The MNIST-shaped MLP as an example, gradient-checked and trained on the CPU reference
summary: examples/mnist_mlp.mspec, 784-64-10 with SGD lr 0.1 for 1,800 steps on synthetic data, in check-examples with a sampled gradient check and a training-IR golden file
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- informed_by: epic:pytorch-10x-mnist
- depends_on: story:observatory-live-event-stream
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Taskfile.yml
- confidence: inferred
  path: crates/modelspec-cli/tests/golden/mnist_mlp.training.ir
- confidence: inferred
  path: crates/modelspec-cli/tests/mnist_mlp.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients.rs
- confidence: inferred
  path: examples/mnist_mlp.mspec
revision: 10
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T03:18:54Z", actor: "human:timo", revision: 8, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "proposed", to: "active", at: "2026-10-22T03:18:54Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "active", to: "implemented", at: "2026-10-22T06:09:00Z", actor: "human:timo", revision: 10, decided_on: {"recorded":{"test_result":1,"review_outcome":2}}}
---
# The MNIST-shaped MLP as an example, gradient-checked and trained on the CPU reference

## Context

`epic:m1-vulkan-execution` takes the nine checkboxes of DESIGN.md § 9 "M1 acceptance criteria" as
its acceptance. The first three concern the language, the compiler and the CPU reference: "A
`.mspec` file defines a linear model or MLP, loss, and SGD without Python", "Rust compiler parses,
checks, builds typed IR, generates gradients and optimizer step" and "CPU reference backend trains
and validates gradients". M0 already meets all three, for linear regression
(`examples/linear_regression.mspec`) and for the 4 → 8 → 1 model the site adds
(`crates/modelspec-docs/models/two_layer_relu.mspec`; `two_layer_relu` in
`crates/modelspec-runtime/tests/gradients.rs`). This story does not claim them as new work.

It exists for two consumers. The GPU stories of this epic need one MLP whose shapes they all run
(`story:m1-vulkan-elementwise`, `story:m1-compile-artifact`, `story:m1-vulkan-train`), and
`epic:pytorch-10x-mnist` (related as `informed_by`) measures the shapes of the MNIST benchmark
example: 784 → 64 ReLU → 10, no bias, f32, MSE, SGD lr 0.1, batch 100, 1,800 steps
(`verification-report:mnist-benchmark-0-1-1`). This story puts that model into `examples/` in the
M0 subset, so the M1 device runs and the 10× work share one file. M1 has no data file (DESIGN.md
§ 12 "Data interface" is open), so the example trains on the M0 synthetic teacher data of
DESIGN.md § 2.3 and § 9 gate 4, not on MNIST.

## What it adds

- `examples/mnist_mlp.mspec`: model `MnistMlp` with `param w1: Tensor<f32, [784, 64]>` and
  `param w2: Tensor<f32, [64, 10]>`, both `xavier_uniform()`; `forward(x: Tensor<f32, [100, 784]>)`
  returns `matmul(relu(matmul(x, w1)), w2)`; a `train` block with `input x`, `target y:
  Tensor<f32, [100, 10]>`, `loss = mse(model.forward(x), y)`, `optimizer = sgd(lr = 0.1)`,
  `steps = 1800`, a fixed `seed` and `device = vulkan(0)`. A header comment says the shapes are
  the benchmark shapes and the data is synthetic until a data interface exists.
- `task check-examples` checks it beside `linear_regression.mspec`; the AGENTS.md line for
  `task check-examples` names it.
- A finite-difference check of the complete example graph at sampled coordinates, and a golden file
  of its training IR.

## Domain relations

None. This story adds a source file and tests; it writes no artifact, checkpoint or device profile
and no type in `spec/`.

## Scope

- `examples/mnist_mlp.mspec` (new).
- `Taskfile.yml` (`check-examples`); `AGENTS.md` (the `task check-examples` line). Both are also
  edited by `story:observatory-live-event-stream` (wave observatory-1); this story is scoped
  against `main` after that wave merges.
- `crates/modelspec-runtime/tests/gradients.rs` (a `mnist_mlp_sampled` test).
- `crates/modelspec-cli/tests/mnist_mlp.rs` (new) and
  `crates/modelspec-cli/tests/golden/mnist_mlp.training.ir` (new).
- Not `crates/modelspec-docs/src/example.rs` or `website/docs/examples/`: the generator pages the
  fixed list `EXAMPLES` (`crates/modelspec-docs/src/example.rs:62`), not every file of `examples/`,
  so a new example gets no page unless it is added there, and this story adds none.
- `CHANGELOG.md`, one line. It is an append-only surface: this story adds its line under
  `## [Unreleased]` and orders against no other story for it.
- `story:m1-inspect-device` edits the State section of `AGENTS.md`; it depends on this story and
  lands after it.

## Acceptance

`examples/mnist_mlp.mspec` is checked by the gate, its gradients match central differences, its
training IR holds both gradients and both SGD updates, and the CPU reference trains it.

1. `task check-examples` exits 0 and runs `modelspec check examples/mnist_mlp.mspec`.
2. `cargo test -p modelspec-runtime --test gradients mnist_mlp_sampled`: at 16 coordinates of each
   of `w1` and `w2`, chosen from a fixed seed and kept away from relu kinks by the existing
   `KINK_MARGIN`, reverse-mode gradients agree with f64 central differences within the existing
   `TOL_F64 = 1e-6` and `TOL_F32 = 1e-3` (`crates/modelspec-runtime/tests/gradients.rs:29-31`).
3. `cargo test -p modelspec-cli --test mnist_mlp training_ir`: `modelspec ir
   examples/mnist_mlp.mspec --stage training` exits 0 and its stdout equals
   `crates/modelspec-cli/tests/golden/mnist_mlp.training.ir` byte for byte; the golden file holds a
   gradient and an SGD update for each of `@MnistMlp.w1` and `@MnistMlp.w2`.
4. `cargo test --release -p modelspec-cli --test mnist_mlp -- --ignored trains_on_cpu`, run by hand
   (a CPU run; the 0.1.1 benchmark measured 19.9 ms per step for these shapes, about 36 s):
   `modelspec train examples/mnist_mlp.mspec --backend cpu` exits 0, prints 2 `init` lines, 37
   `step` lines (every 50th step and the last, `LOSS_EVERY` in `crates/modelspec-cli/src/train.rs`)
   and 2 `param` lines, and the loss on the last `step` line is below the step-0 loss.
5. `task ci` exits 0, `task docs-check` included; no Python, PyTorch or CUDA enters the tree.

## Out of scope

- MNIST itself: `train --data` (the `story:bert-train-data-binding` lane of
  `epic:pytorch-10x-mnist`).
- Modules and generics (`examples/tiny_mlp.mspec`, DESIGN.md § 11 iteration 3).
- A recorded 1,800-step CPU training transcript on the site.
- Anything on the GPU: `story:m1-vulkan-train` runs this example there.
