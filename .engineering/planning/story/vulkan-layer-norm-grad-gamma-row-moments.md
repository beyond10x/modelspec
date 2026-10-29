---
format: aep.planning-md/3
id: story:vulkan-layer-norm-grad-gamma-row-moments
kind: story
status: implemented
title: LayerNormGradGamma reads each row moments once
summary: A per-row moments dispatch feeds the column kernel, so no workgroup recomputes every row mean and rstd; bits unchanged.
relations:
- decomposes: epic:bert-tiny-parity-with-eager
- serves: vision:modelspec
scope:
- confidence: inferred
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: inferred
  path: crates/modelspec-core/src/schedule.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
- confidence: inferred
  path: crates/modelspec-runtime/src/vulkan/stepwise.rs
revision: 8
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T06:02:11Z", actor: "human:timo", revision: 6}
- {from: "proposed", to: "active", at: "2026-10-29T06:02:12Z", actor: "human:timo", revision: 7}
- {from: "active", to: "implemented", at: "2026-10-29T10:15:13Z", actor: "human:timo", revision: 8, decided_on: {"recorded":{"test_result":1}}}
---
# LayerNormGradGamma reads each row moments once

## Context

story:vulkan-column-sums-coalesced (d2fdbc18, merged ff70c0c8) split the column sums over
workgroups and staged rows in workgroup memory. SumRows dropped to about 0.37 of its time, but
LayerNormGradGamma[4096,128] only to 0.52 (3714 -> 1917 us/step on a GPU then held in P3): every
workgroup recomputes the mean and rstd of every row before it can form xhat, so that work is
repeated per workgroup and waits on memory and one barrier per 16-column block. Coordinator
decision D153 (2026-10-29) adds this unit to the eager-parity wave.

## Outcome

LayerNormGradGamma becomes two dispatches: a per-row moments kernel that writes each row mean and
rstd, by the CPU recipe, into an f32[n, 2] buffer, then the column kernel that reads them instead of
recomputing. Each column still adds its terms in ascending row order from -0.0, so the bits stay
the CPU bits. Estimate from the colsums lane: about 25 us per 4096-row site for the moments.

## Acceptance

- A device test, red first: the plan shows the moments dispatch, a barrier and the column dispatch
  for every LayerNormGradGamma site; every gradient element bit-equal to the CPU on [4097, 129],
  [1, 1], [65, 3] and the BERT-tiny shapes, under Vulkan validation.
- bert_tiny_vulkan_bits and vulkan_bert_tiny_same_bits keep their recorded bits.
- Interleaved A/B, 3 pairs, display awake (AGENTS.md), clocks sampled: LayerNormGradGamma time per
  step and steady us/step after/before.
