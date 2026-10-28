---
format: aep.planning-md/3
id: story:decider-small-operators
kind: story
status: rejected
title: 'Small operators: reshape, concat, smooth_l1, sqrt, abs, masked cross_entropy, negative literals'
summary: Seven source-language additions, each gradient-checked and with a Vulkan kernel held to the CPU reference.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/check_small_ops.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/autodiff.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/tensor.rs
- confidence: cited
  path: crates/modelspec-core/src/ops.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/schedule.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/operators.rs
- confidence: cited
  path: crates/modelspec-observatory/schema
- confidence: cited
  path: crates/modelspec-runtime/schema
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit.rs
- confidence: cited
  path: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/gradients_small_ops.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/vulkan_small_ops.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T12:12:30Z", actor: "human:timo", revision: 5}
- {from: "proposed", to: "rejected", at: "2026-10-28T13:32:23Z", actor: "human:timo", revision: 7}
---
# Small operators: reshape, concat, smooth_l1, sqrt, abs, masked cross_entropy, negative literals

**Superseded** (2026-10-28, after the plan critics): split into three stories that land one after
the other, each with its own gradient tests and Vulkan kernels. Implement those, not this one. The
move of this story to `rejected` is the coordinator's; it has not been made.

| Addition | Story |
|---|---|
| `reshape`, last-axis `concat` | `story:decider-shape-operators` |
| `smooth_l1`, masked `cross_entropy` | `story:decider-loss-operators` |
| `sqrt`, `abs`, negative float literals | `story:decider-elementwise-operators` |

Each of the three decomposes `epic:decider-training-support`, serves `vision:modelspec` and
`supersedes` this story. The outcome, context, acceptance and out-of-scope lists that were here
moved into them, with the comparison named per operator and a code named per refusal; the landing
order is in the epic's § Landing order.
