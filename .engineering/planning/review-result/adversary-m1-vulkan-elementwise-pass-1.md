---
format: aep.planning-md/3
id: review-result:adversary-m1-vulkan-elementwise-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-vulkan-elementwise
relations:
- reviews: story:m1-vulkan-elementwise
revision: 1
---
# Adversary pass 1 on story:m1-vulkan-elementwise

- Worktree `impl/m1-vulkan-elementwise` at `6937dac` plus the uncommitted unit.
- Verdict NEEDS-CHANGE. 10 tests in `crates/modelspec-runtime/tests/adversary_elementwise.rs`, 6 red.
- Device results on the RTX 3090 were correct; three suite checks could not fail for the defects
  they target, and two public APIs accepted inputs with undefined device behaviour.
- Red: tolerance pair ignored by `Tolerance::accepts`; SGD guard words miss the `<=` bounds mutant;
  relu-grad inputs miss the `>=` mutant; a pipeline with fewer bindings than the module declares
  is created (VUID-VkComputePipelineCreateInfo-layout-07988 under validation); a dispatch past the
  bound buffers is recorded; 24 subnormal-operand mismatches against the CPU interpreter.
- Not broken: rank-0 operand on either side, zero-size buffers, 2D workgroup indexing, NaN/inf/±0
  propagation with normal operands, `spirv-val` absent, validation clean, buffer alignment.

```findings
[{"file": "crates/modelspec-runtime/tests/common/mod.rs", "line": 61, "category": "acceptance", "severity": "warning", "verdict": "CONFIRMED", "origin": "introduced", "message": "Tolerance::accepts ignores the atol/rtol pair the per-operator lookup returns and always applies the global constants, so the per-operator tolerance is only a label"}, {"file": "crates/modelspec-runtime/tests/vulkan_elementwise.rs", "line": 159, "category": "mutant", "severity": "warning", "verdict": "NEEDS-CHANGE", "origin": "introduced", "message": "the SGD guard words sit only after the parameter, so a <= bounds-check mutant leaves them unchanged at every non-multiple-of-64 count; the gradient must carry guard words too"}, {"file": "crates/modelspec-runtime/tests/vulkan_elementwise.rs", "line": 55, "category": "mutant", "severity": "warning", "verdict": "NEEDS-CHANGE", "origin": "introduced", "message": "both relu-grad operands are zero at the same positions, so the x == 0 case cannot catch a >= comparison mutant"}, {"file": "crates/modelspec-runtime/src/vulkan/pipeline.rs", "line": 72, "category": "boundary", "severity": "warning", "verdict": "INFEASIBLE", "origin": "introduced", "message": "Device::pipeline takes the binding count from the caller instead of the module and creates a pipeline whose layout lacks a binding the shader uses, with no typed error"}, {"file": "crates/modelspec-runtime/src/vulkan/dispatch.rs", "line": 140, "category": "boundary", "severity": "warning", "verdict": "INFEASIBLE", "origin": "introduced", "message": "Recording::dispatch records an element count larger than the bound buffers, which writes out of bounds with robustBufferAccess off"}, {"file": "crates/modelspec-runtime/src/spirv/emit.rs", "line": 261, "category": "boundary", "severity": "note", "verdict": "INFEASIBLE", "origin": "introduced", "message": "without DenormPreserve the RTX 3090 flushes subnormals, so relu-grad of a positive subnormal and Mul(inf, subnormal) fall outside the gate-5 tolerance"}, {"file": "crates/modelspec-runtime/src/vulkan/memory.rs", "line": 43, "category": "judgement", "severity": "note", "verdict": "INFEASIBLE", "origin": "introduced", "message": "read_f32 takes &self, so safe code can read a staging buffer while a pending submission still writes it"}]
```
