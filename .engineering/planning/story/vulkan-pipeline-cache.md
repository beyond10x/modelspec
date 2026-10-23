---
format: aep.planning-md/3
id: story:vulkan-pipeline-cache
kind: story
status: draft
title: A Vulkan pipeline cache
relations:
- informed_by: story:m1-compile-artifact
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/pipeline.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
revision: 2
---
# A Vulkan pipeline cache

## Why

Review of 0.7.1 (finding 11), at main 496e98e. The runner creates every pipeline at startup
(`crates/modelspec-runtime/src/vulkan/runner.rs:471-478`) with `PipelineCache::null()`
(`crates/modelspec-runtime/src/vulkan/pipeline.rs:208`). story:m1-compile-artifact deferred a cache on
disk keyed by `KernelCacheKey` (its line 162). This is startup time only.

Outcome: a second run of the same artifact creates its pipelines from a cache.

## Acceptance

- **second-run-pipeline-time**: `--profile`'s pipeline-creation time of a second run is reported
  against the first, and is lower.
- **cache-keyed**: a cache entry is keyed by `KernelCacheKey` plus the device and driver, and a
  mismatching entry is ignored, not used.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/vulkan/pipeline.rs`
- `crates/modelspec-runtime/src/vulkan/runner.rs`

## Out of scope

- Compile time, which story:vulkan-train-compiles-once holds.
- Step time; the cache does not change the timed loop.
