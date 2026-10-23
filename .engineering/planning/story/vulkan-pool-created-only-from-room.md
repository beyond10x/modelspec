---
format: aep.planning-md/3
id: story:vulkan-pool-created-only-from-room
kind: story
status: draft
title: The descriptor pool is created only from its room
revision: 1
---
# The descriptor pool is created only from its room

## Why

`crates/modelspec-runtime/src/vulkan/pool.rs` builds the pool's create-info through `pool_info`,
which host tests read. A `create_pool` that built its create-info inline would pass every host
test (adversary passes 1–2 of story:vulkan-descriptor-pool-counts-writes,
review-result:adversary-descriptor-pool-passes-1-2).

## Acceptance

- `create_pool` takes only the value `pool_info` returns, in a type no other code can build, so an
  inline create-info does not compile.
- `task ci` exits 0.
