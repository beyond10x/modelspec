---
format: aep.planning-md/3
id: review-result:adversary-m1-vulkan-reduce-matmul-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-vulkan-reduce-matmul
relations:
- reviews: story:m1-vulkan-reduce-matmul
revision: 1
---
# Adversary pass 1, story:m1-vulkan-reduce-matmul

Verdict: CONFIRMED. Target: uncommitted tree of impl/m1-vulkan-reduce-matmul on base 23adba4.
Added tests: crates/modelspec-runtime/tests/adversary_reduce_matmul.rs (device),
crates/modelspec-runtime/tests/adversary_reduce_matmul_spirv.rs (CI). Executed 1062 -> 1066,
red 0.

RTX 3090, validation on: class tolerance; 292 cases run twice, 296494 elements, 0 failures,
0 validation messages. `task ci` exit 0, 1066 passed, 0 failed.

Mutation sweep: 11 hand mutants of emit.rs. M11 (stage-1 `started` guard dropped, spare
workgroups of a [65535, 2, 1] grid write past the partials buffer) survives the implementor's
suite and is killed by `adversary_stage1_writes_no_partial_past_the_last_block`, which the unit
keeps. The other 10 are killed by the implementor's suite.

```findings
[
{"file":"crates/modelspec-runtime/src/spirv/emit.rs","line":924,"category":"mutant","severity":"warning","verdict":"CONFIRMED","origin":"introduced","message":"Dropping the stage-1 started guard lets the 65533 spare workgroups of a [65535, 2, 1] grid write past the partials buffer, and the unit's own suites stay green because no test guards the partials buffer or reduces more than 4,194,240 elements; adversary_stage1_writes_no_partial_past_the_last_block kills it."},
{"file":"crates/modelspec-runtime/src/spirv/emit.rs","line":204,"category":"judgement","severity":"note","verdict":"INFEASIBLE","origin":"undecided","message":"KernelCacheKey hard-codes fast_math false and deterministic true instead of reading the artifact emit_kernels receives; nothing in modelspec-core produces another setting today."}
]
```
