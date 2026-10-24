---
format: aep.planning-md/3
id: story:fast-math-parallel-loss-sum
kind: story
status: draft
title: A multi-workgroup loss sum under fast_math
relations:
- serves: vision:modelspec
- depends_on: story:fast-math-mode
- depends_on: story:vulkan-loss-slots-written-by-kernel
revision: 1
---
# A multi-workgroup loss sum under `fast_math`

## Why

Split from story:fast-math-mode on 2026-10-24 (decision D12 on `.engineering/waves/perf-1.md`).
The `mse` loss sum runs in one workgroup (`[1, 1, 1]`,
`crates/modelspec-core/src/schedule.rs:456`), invocation 0 adding the terms in row-major order
(DESIGN.md § 5.2). At batch 8192 the loss site takes 375.5 µs of a 15,731 µs profiled step
(`benchmarks/raw/2026-10-24-tiling/after-profile-b8192-h4096.txt`, step 4). ADR-008 permits the
sum to spread over many workgroups in two stages whose order the shape alone fixes, as for the
two-stage reductions of § 5.2. story:vulkan-loss-slots-written-by-kernel changes the same kernel
to write the step's loss slot, so this story follows it.

Outcome: under `fast_math` the loss-sum kernel runs over more than one workgroup in two fixed
stages; without the flag it keeps `[1, 1, 1]` and its order.

## Acceptance

- **parallel-loss-sum**: under `fast_math` the loss-sum kernel of the MNIST plan and of the sweep
  plan at b8192 h4096 has more than one workgroup in `ir --stage kernel`; without it, it keeps
  `[1, 1, 1]`.
- **order-fixed-by-shape**: the two-stage order is a function of the shape alone; two runs of one
  `fast_math` artifact on one device give the same bits, and DESIGN.md § 5.2 states the order.
- **tolerance**: the loss curve under `fast_math` stays within ADR-008's tolerance of the CPU
  oracle on `examples/mnist_mlp.mspec` and the sweep source at b8192 h4096.
- **default-unchanged**: with `fast_math` off, the artifact hash and the loss bits are unchanged.
- **loss-site-reported** (reported, not gated): the loss site's `mean_us` from `--profile` at
  b8192 h4096 before and after.
- `task ci` exits 0.

## Scope

- `crates/modelspec-core/src/schedule.rs`
- `crates/modelspec-core/src/ir/kernel.rs`
- `crates/modelspec-runtime/src/spirv/emit.rs`
- `DESIGN.md`
