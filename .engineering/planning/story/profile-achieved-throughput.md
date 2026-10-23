---
format: aep.planning-md/3
id: story:profile-achieved-throughput
kind: story
status: draft
title: Achieved GFLOP/s and GB/s in train --profile
relations:
- decomposes: epic:transformer-vs-pytorch-cuda
revision: 2
---
# Achieved GFLOP/s and GB/s in train --profile

## Context

`epic:transformer-vs-pytorch-cuda`. `train --profile` prints µs per dispatch site and per step
(website/docs/reference/cli.md:145) but no arithmetic or memory figure, so nobody can say how much
of the GPU a run uses. The kernel plan knows every dispatch's shapes, so it can count FLOP and bytes
moved without a vendor profiler.

## Acceptance

- Each kernel-plan dispatch of the matmul, elementwise and reduction kinds carries a FLOP count and
  a byte count (operand reads plus result writes), computed from its shapes; a unit test checks both
  for one matmul, one elementwise and one reduction dispatch against hand-computed values.
- A dispatch kind with no counting rule prints `uncounted` in place of its figures, never 0, and
  the per-step total names how many sites it left out; a unit test covers this.
- `train --backend vulkan --profile` prints, per dispatch site, achieved GFLOP/s and GB/s, and per
  step the total.
- `verification-report:mnist-profile-throughput-<date>` holds that output for
  `examples/mnist_mlp.mspec` on the RTX 3090.
- `modelspec ir --stage kernel` shows the two counts per dispatch.
- DESIGN.md states the counting rule, and the CLI reference is regenerated.
