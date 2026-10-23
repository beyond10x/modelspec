---
format: aep.planning-md/3
id: story:vulkan-descriptor-pool-counts-writes
kind: story
status: active
title: The Vulkan descriptor pool counts every bound buffer
relations:
- serves: vision:modelspec
- decomposes: epic:m1-vulkan-execution
revision: 4
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
- The descriptor arena counts the descriptors it binds and refuses a bind past the pool's size with a named error, so an undersized pool fails on every driver; a host-only test reaches the refusal.
- `task ci` exits 0.

## Changed acceptance (2026-10-23)

The first draft asked for a device test that fails on any validation message. The MNIST device test already does that, and its run on the undersized pool was green (`.engineering/evidence/story/m1-vulkan-train/20261022T102152Z-002-f5b15754db80.json`): standard validation on the RTX 3090 does not report the shortfall; only best-practices validation does. The host-side refusal replaces it (approval-record:benchmark-wave-scope D13).
