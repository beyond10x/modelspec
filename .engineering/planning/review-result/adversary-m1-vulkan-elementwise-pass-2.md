---
format: aep.planning-md/3
id: review-result:adversary-m1-vulkan-elementwise-pass-2
kind: review-result
status: active
title: Adversary pass 2, story:m1-vulkan-elementwise
relations:
- reviews: story:m1-vulkan-elementwise
revision: 1
---
# Adversary pass 2 on story:m1-vulkan-elementwise

- Worktree `impl/m1-vulkan-elementwise` at `6937dac` plus the unit after fix round 1.
- Verdict INFEASIBLE: 5 device tests in `crates/modelspec-runtime/tests/adversary_elementwise_2.rs`,
  4 red; no production caller reaches them yet.
- Red: a 64-element dispatch over a 1-element or 2-byte tensor binding (input or output) is
  recorded as Ok; a module bound in descriptor set 1 gets a set-0 pipeline.
- Mutants against the fixed suite: the `<=` bounds mutant reddens the suite (80 guard words, 4 in
  SGD cases); relu-grad `>=` reddens it (21 elements); `Tolerance::accepts` on the global pair
  reddens a test in `task ci`; a no-op subnormal flush reddens `vulkan_subnormal`.
- Not broken: the implementor's edits to the pass-1 adversary file relax nothing; duplicated and
  reordered bindings; the `read_f32` compile-fail doctest.

```findings
[{"file": "crates/modelspec-runtime/src/vulkan/dispatch.rs", "line": 159, "category": "boundary", "severity": "warning", "verdict": "INFEASIBLE", "origin": "introduced", "message": "the elements > 1 exemption records a 64-element dispatch over a 1-element or 2-byte tensor binding, input or output, as Ok instead of CountPastBuffer"}, {"file": "crates/modelspec-runtime/src/vulkan/pipeline.rs", "line": 79, "category": "boundary", "severity": "note", "verdict": "INFEASIBLE", "origin": "introduced", "message": "declared_bindings ignores DescriptorSet, so a module bound in set 1 yields a set-0 pipeline that only the validation layer flags"}, {"file": "crates/modelspec-runtime/tests/common/mod.rs", "line": 36, "category": "mutant", "severity": "note", "verdict": "CONFIRMED", "origin": "introduced", "message": "SGD_GRADIENT_GUARDED=false leaves task ci and the device suite green and reddens only an ignored adversary test; the knob has no use for false"}, {"file": "DESIGN.md", "line": 511, "category": "contract-drift", "severity": "note", "verdict": "CONFIRMED", "origin": "introduced", "message": "gate 5 says every case checks 16 guard words and flushes only on preserve = false, while the chain and subnormal tests check no guards and the test flushes on != Some(true)"}]
```
