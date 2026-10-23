---
format: aep.planning-md/3
id: story:vulkan-descriptor-pool-counts-writes
kind: story
status: active
title: The Vulkan descriptor pool counts every bound buffer
relations:
- serves: vision:modelspec
- decomposes: epic:m1-vulkan-execution
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-23T18:08:55Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-23T18:08:55Z", actor: "human:timo", revision: 3}
---
# The Vulkan descriptor pool counts every bound buffer

## Context

Found by the whole-codebase review of 2026-10-23 (blocker, confirmed by a host-only probe).
`crates/modelspec-runtime/src/vulkan/runner.rs:541` sizes the descriptor pool at `reads + 1`
storage-buffer descriptors per dispatch, while each set binds `reads + writes`. A `Relu` with
`writes_pre_activation: true` binds two writes, so `examples/mnist_mlp.mspec` allocates 39
descriptors from a pool of 38. NVIDIA's driver over-provisions and hides it; a driver that
accounts exactly can refuse the allocation with `VK_ERROR_OUT_OF_POOL_MEMORY`. Shipped since 0.4.0.

## Acceptance

- The pool's storage-buffer count is the sum over dispatch steps of `reads + writes`; a host-only
  unit test computes it for the MNIST kernel plan (39) and for a plan with a two-write step.
- A test that runs under `MODELSPEC_VK_VALIDATION=1` on the device reads the validation messages
  and fails on any, so an undersized pool is caught where validation reports it.
- `task ci` exits 0.
