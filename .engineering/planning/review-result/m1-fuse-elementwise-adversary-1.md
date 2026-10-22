---
format: aep.planning-md/3
id: review-result:m1-fuse-elementwise-adversary-1
kind: review-result
status: active
title: Adversary review of m1-fuse-elementwise at bdd039d
relations:
- reviews: story:m1-fuse-elementwise
revision: 1
---
# Adversary review of m1-fuse-elementwise at bdd039d

Verdict: NEEDS-CHANGE, documentation only. Base and new binary give identical `train --backend
vulkan` output on 10 models; no change of bits, schedule or barrier found. Cases 892 to 900
executed, 8 added, 4 red (all documentation contracts).

Not broken: same rounded operations in the same order with `NoContraction` on NaN, -0.0, a
subnormal-flushing model, `p - (lr * g)` and folded-constant chains (`0.7*0.3`, lr 0.13); an f64
fold mutant gives the same results as the f32 fold; a ReLU whose MatMul has 0, 1 or several other
readers, two ReLUs on one MatMul, a transposed matmul, a ReLU as or after the prediction, the
forward plan and `--profile`; no buffer read before its writer's barrier in 6 models and both
plans. Device tests the implementor did not run, all green: `vulkan_profile` 7,
`vulkan_profile_adversary` 3, `adversary_vulkan_train_chunks` 2, `vulkan_device` 1, CLI
`train_vulkan` 6, `train_profile_adversary` 2.

Residual, not reachable on the RTX 3090: on a device that reports the float-control features, the
ReLU epilogue runs under `DenormPreserve` and `SignedZeroInfNanPreserve` while the unfused ReLU ran
in default modes.

```findings
- file: website/docs/guides/inspect-a-run.md
  line: 54
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "Quoted kernel lines k0 d68423b2 and k3 d588cbc2 and 'step position 6 ... is k3' (line 117) no longer match `ir --stage kernel examples/mnist_mlp.mspec` (k0 is 0f20c50b with epilogue Relu, the loss sum is k2); red case adversary_guide_kernel_lines.rs"
- file: spec/domains/artifact.yaml
  line: 227
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "'In M1 each value has its own buffer, except an SGD update' is false: lr*g and an unkept pre-activation have no buffer (MNIST 17 buffers, rule needs >= 19); red case every_value_has_its_own_buffer_as_the_artifact_spec_states, green at base"
- file: DESIGN.md
  line: 164
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "§ 3.1 opens with 'one buffer and one view per value ... except that the SGD update' and later says the pre-activation has no buffer; the opening clause needs the fusion exceptions"
- file: crates/modelspec-core/src/schedule.rs
  line: 10
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "Module doc says Relu and ReluGrad of compile-time scalars fold; fold() folds only Add/Sub/Mul, so relu(mul(2.0,3.0)) keeps 'elementwise Relu f32 (6.0)'. Line 22 still says 'Every value has its own buffer'"
- file: CHANGELOG.md
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "Unreleased fusion line does not say a scalar stays unfolded when a reader takes no immediate (sum(mul(2.0,3.0)) keeps its Mul kernel)"
- file: crates/modelspec-runtime/tests/vulkan_elementwise.rs
  line: 437
  category: weak-test
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "The device fold check uses only 1.0*0.002, an exact product, so a fold that rounds differently from the device would pass; DESIGN § 3.1 cites it as the check of the folded value"
- file: crates/modelspec-runtime/src/spirv/emit.rs
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: "On a device that reports the three float-control features, the relu epilogue runs under DenormPreserve/SignedZeroInfNanPreserve while the unfused relu ran in default modes, so 'same bits' may not hold there; the RTX 3090 reports none of these features and cannot show it"
```
