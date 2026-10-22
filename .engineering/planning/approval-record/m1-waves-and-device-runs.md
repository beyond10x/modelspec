---
format: aep.planning-md/3
id: approval-record:m1-waves-and-device-runs
kind: approval-record
status: draft
title: M1 waves and RTX 3090 device runs approved
relations:
- serves: vision:modelspec
revision: 1
---
# M1 waves and device runs approved

- Relayed by b10x-chef-orchestrator on 2026-10-22, after release 0.3.0, quoting the operator in
  chat at 07:4x: "There is no cost. I mean, my GPU is running here. I mean, don't be shy. You can
  dispatch both stories if you want. I mean, just keep some headroom later for isolated testing
  ... Speed up a bit now ... use more subagents".
- Chef's direction: move the M1 stories `m1-inspect-device`, `m1-vulkan-elementwise`,
  `m1-vulkan-reduce-matmul` and `m1-vulkan-train` to active and run the M1 wave; device runs on the
  RTX 3090 are approved for the M1 stories and the MNIST benchmark, with no per-run ask.
- In parallel: the CLI benchmark lane and a PyTorch multi-thread CPU baseline, PyTorch kept outside
  the repository (`epic:pytorch-10x-mnist`, Measurement).
- Headroom: a timed benchmark run has the GPU and its CPU core to itself; nothing else runs on the
  GPU during a timed run, and the report says so.
- Fan-out: one implementor per story in its own worktree, adversary passes in parallel; builds
  through `b10x-build-slot`.
- Source: the cross-session message from b10x-chef-orchestrator; the operator's words are as chef
  quoted them, not read from the chat directly.
