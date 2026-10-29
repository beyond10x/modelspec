---
format: aep.planning-md/3
id: story:vulkan-pipeline-startup
kind: story
status: implemented
title: 'Pipeline creation at startup: cached, parallel, deduplicated'
relations:
- decomposes: epic:bert-tiny-step-tenth
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T21:54:45Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-10-28T21:54:45Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-10-29T07:18:18Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# Pipeline creation at startup: cached, parallel, deduplicated

## Outcome

A Vulkan run creates its compute pipelines through a `VkPipelineCache` persisted on disk (keyed by
device UUID and driver version, under the user cache directory, never the repository), creates
them in parallel, and creates one pipeline per distinct SPIR-V module. A warm start of
`examples/bert_tiny.mspec` spends under 1 s in pipeline creation.

## Context

`pipeline.rs:412-424` creates 113 pipelines one by one with `PipelineCache::null()`; the profile
reports `pipeline_creation_us` 6.3 s to 15.1 s per run. 112 distinct module signatures reduce to
37 once shapes are stripped (`emit.rs:255-275`). Explorer notes: scratch `explore/e3.md`.

## Scope

`crates/modelspec-runtime/src/vulkan/pipeline.rs`, `stepwise.rs`, `runner.rs`, `spirv/emit.rs`
module keys, DESIGN.md § 8.

## Acceptance

- `pipeline_creation_us` cold and warm, before and after, on `examples/bert_tiny.mspec`.
- A corrupt or foreign cache file is ignored with a warning, and the run's bits are unchanged.
