---
format: aep.planning-md/3
id: story:m1-fuse-elementwise
kind: story
status: active
title: 'Fewer dispatch groups per step: ReLU into its matmul, scale into the SGD update, constants folded'
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
- depends_on: story:m1-vulkan-profile
- depends_on: story:m1-loss-sum-staged
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: inferred
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/dispatch.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_elementwise.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_train.rs
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T11:18:05Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-22T11:18:05Z", actor: "human:timo", revision: 5}
---
# Fewer dispatch groups per step: ReLU into its matmul, scale into the SGD update, constants folded

## Context

`train --backend vulkan --profile` (`story:m1-vulkan-profile`, wave m1-e) on
`examples/mnist_mlp.mspec`, 72,000 steps, RTX 3090 at P0 1951 MHz, profiled step 90.3 µs. The step
plan has 18 dispatches in 11 barrier groups. Three groups hold one small elementwise kernel each,
all on the critical path:

| position | kernel | mean µs | what it does |
|---|---|---|---|
| 2 | `k1` Relu | 1.87 | `relu(x @ W1)`, alone between two barriers |
| 22 | `k14` Mul | 2.34 | `0.1 * dW1`, alone between two barriers |
| 27 | `k17` sgd_update | 1.87 | `W2 - 0.1 * dW2`, alone after `k16` |

Each one alone is under 5 % of the profiled step (2.0 %, 2.5 %, 2.0 %), so wave m1-e drafted them
as one story: together about 6.1 µs, 6.6 %, plus three barriers whose cost the profile does not
time. Inferred, not measured: an unprofiled dispatch costs less than its profiled mean, so the saving
may be smaller than 6.1 µs.

`k5` (position 8) computes `1.0 * 0.002` on the device every step; DESIGN.md § 5.4 already says the
gradient scale of `mse` is folded to a constant at compile time.

## What it changes

- Schedule (`schedule::schedule`): a `Relu` whose input is a `MatMul` output is emitted as one
  `MatMul` kernel with a ReLU epilogue (compare and select on the rounded sum). Where the
  pre-activation has another reader (on the example `k11` ReluGrad reads `b4`), the fused kernel
  writes both the pre-activation and the activation; otherwise only the activation. The SGD scale
  `lr * grad` and the subtraction are one `sgd_update` kernel reading the gradient directly; a `Mul`
  of two compile-time scalars is folded into the immediate of its reader.
- On the MNIST example: dispatches per step 18 → 14 (k1, k5, k14, k16 gone), barrier groups 11 → 8.
- The bits do not change: the ReLU epilogue is the same compare and select on the same rounded
  matmul output, and `p - (lr * g)` keeps its two rounded operations in the same order with
  `NoContraction`.
- DESIGN.md § 3.1 (one dispatch per computing op, except these fusions) and the revision line.

## Scope

- `crates/modelspec-core/src/schedule.rs`, `crates/modelspec-core/src/ir/kernel.rs` (an epilogue on
  `MatMul`, the scaled `SgdUpdate`), `crates/modelspec-core/src/ir/print.rs`,
  `crates/modelspec-core/src/ir/verify.rs` if the plan verifier names kernel shapes.
- `crates/modelspec-runtime/src/spirv/emit.rs`, `crates/modelspec-runtime/src/vulkan/dispatch.rs`.
- Tests: `crates/modelspec-core` schedule tests, `crates/modelspec-runtime/tests/vulkan_train.rs`,
  `vulkan_elementwise.rs`, `spirv_val.rs`, and the `ir --stage kernel` snapshots of the CLI tests.
- `DESIGN.md` § 3.1 and the revision line; `CHANGELOG.md`.
- Shares `emit.rs`, `schedule.rs`, `kernel.rs`, DESIGN.md and CHANGELOG.md with
  `story:m1-loss-sum-staged`; the wave merges that one first, and this one rebases on it.

## Acceptance

Four fewer dispatches and three fewer barrier groups per step, the same results.

1. `modelspec ir examples/mnist_mlp.mspec --stage kernel`: 14 dispatches in the step plan, no
   kernel whose inputs are all compile-time scalars, the ReLU as an epilogue of `k0`, one
   `sgd_update` per parameter reading the gradient. In `task ci`.
2. Operator run on the RTX 3090: `vulkan_train.rs` and `vulkan_elementwise.rs` device tests pass;
   the parameters after 1,800 steps equal those of the base commit's GPU run bit for bit on the same
   device, and match the CPU oracle within the existing per-operator tolerances.
3. Schedule tests: where the pre-activation has a second reader the fused kernel writes both
   buffers and that reader reads the pre-activation; where it has none, the fused kernel writes the
   activation only and no pre-activation buffer is allocated.
4. `train --backend vulkan --stats` on the example prints `dispatches per step: 14`.
5. Steady-state µs per step by the ADR-007 timing method, before and after on the same device:
   median, range and clock reported.
6. `task ci` exits 0.

## Out of scope

- Fusing the backward ReLU gradient into its matmul (`k11` shares its group with longer kernels, so
  the profile shows no saving).
- Transposes read as strided operands, tiled matmul, the whole epoch in one command buffer (m1-f).
