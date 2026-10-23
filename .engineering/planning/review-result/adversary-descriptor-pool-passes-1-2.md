---
format: aep.planning-md/3
id: review-result:adversary-descriptor-pool-passes-1-2
kind: review-result
status: active
title: 'Adversary passes 1-2: acceptance replaced, bind admission made unskippable'
relations:
- reviews: story:vulkan-descriptor-pool-counts-writes
revision: 1
---
# Adversary passes 1 and 2 on story:vulkan-descriptor-pool-counts-writes

Pass 1 on d7476c6 (base c84937f): no break in the code. The second acceptance line (a device test
that fails on any validation message) was not met and could not be: the MNIST device test already
drains validation messages and was green on the undersized pool. The acceptance was changed to a
host-side refusal (approval-record:benchmark-wave-scope D13, story revision 4). The two-write test
recomputed the formula; it now asserts literals.

Pass 2 on 249a3e3: deleting the room check from `DescriptorArena::bind` left all 29 host Vulkan
tests green. Fixed in 1ecbc39: one `admit` function returns a token `allocate_set` requires, so
the deletion no longer compiles, and the pool's create-info is built from the room through
`pool_info`, which host tests read (count −1 and `max_sets` +1 both fail 2 tests).

Attacked without a break: the second pool site in `checked_dispatch` (one exact pool per set), a
refused bind consuming room, u32 overflow, an empty set on a full pool, room versus set-layout size.

Left: a `create_pool` that builds its create-info inline instead of through `pool_info` would pass
every host test.

```findings
- file: crates/modelspec-runtime/tests/vulkan_train.rs
  line: 194
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: no device test met the validation acceptance line, and validation was green on the undersized pool; acceptance replaced by D13
- file: crates/modelspec-runtime/src/vulkan/runner.rs
  line: 1131
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the two-write test compared descriptors() with a copy of its own formula
- file: crates/modelspec-runtime/src/vulkan/dispatch.rs
  line: 265
  category: mutant
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: deleting the room check in bind left all 29 host Vulkan tests green
```
