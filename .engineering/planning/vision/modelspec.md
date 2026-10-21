---
format: aep.planning-md/3
id: vision:modelspec
kind: vision
status: approved
title: 'ModelSpec: a declarative differentiable-program compiler with an owned Vulkan runtime'
revision: 4
transitions:
- {from: "draft", to: "in_review", at: "2026-10-21T08:41:05Z", actor: "human:timo", revision: 2}
- {from: "in_review", to: "approved", at: "2026-10-21T08:41:05Z", actor: "human:timo", revision: 3}
---
# ModelSpec: specify what to compute, compile how to run it

## Objective

A versioned, declarative language and Rust compiler for differentiable tensor programs. A whole
training step (forward, loss, backward, optimizer) is compiled and run through an independently
owned Rust runtime, first on an NVIDIA RTX 3090 through Vulkan compute and SPIR-V, with no Python,
PyTorch or CUDA API/toolkit dependency (DESIGN.md § 0).

## Measures

- M0: linear regression trains correctly on the CPU reference interpreter, and a two-layer relu
  graph passes finite-difference gradient checks (DESIGN.md § 1, § 11 iteration 1). Module-based
  MLPs such as `examples/tiny_mlp.mspec` arrive with § 11 iteration 3.
- M1: the first end-to-end f32 training run on the RTX 3090 through Vulkan meets the nine
  acceptance criteria of DESIGN.md § 9.

## Constraints

The seven non-negotiable invariants of DESIGN.md § 0. Correctness and coherent architecture come
before speed; near-optimal performance is a long-term goal, not a v1 guarantee.
