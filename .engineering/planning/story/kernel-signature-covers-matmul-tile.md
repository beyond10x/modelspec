---
format: aep.planning-md/3
id: story:kernel-signature-covers-matmul-tile
kind: story
status: active
title: A matmul kernel's signature covers every tile parameter
relations:
- decomposes: epic:pytorch-10x-mnist
- serves: vision:modelspec
- informed_by: story:m1-f-weight-grad-matmul-columns
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: inferred
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir
- confidence: cited
  path: crates/modelspec-core/src/ir/kernel.rs
- confidence: inferred
  path: crates/modelspec-core/tests/kernel_plan.rs
- confidence: inferred
  path: website/docs/guides/inspect-a-run.md
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-23T09:09:50Z", actor: "human:timo", revision: 4}
- {from: "proposed", to: "active", at: "2026-10-23T09:09:50Z", actor: "human:timo", revision: 5}
---
# A matmul kernel's signature covers every tile parameter

## Context

`Kernel::signature` (`crates/modelspec-core/src/ir/kernel.rs:359`) leaves out a matmul's tile
depth and ahead. Two modules that differ only in those parameters therefore print the same kernel
id in `train --profile` and `modelspec ir --stage kernel` (adversary note, wave m1-g). This fix was
the third bullet of `story:m1-f-weight-grad-matmul-columns`. That story was archived when its
mechanism measured slower (`verification-report:k11-multicol-2026-10-23`), and its own acceptance
could not pass, so this story carries the fix alone.

## Scope

- `crates/modelspec-core/src/ir/kernel.rs` (the signature and a `Display` for `MatMulTile`).
- Tests and goldens that pin kernel ids: `crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`,
  `crates/modelspec-core/tests/kernel_plan.rs`, and any other test the id change breaks.
- `website/docs/guides/inspect-a-run.md` where it shows a kernel id, `CHANGELOG.md`, and DESIGN.md
  § 5.2 if it states what a signature covers.

## Acceptance

1. A unit test builds two matmul kernels that differ only in tile depth, and two that differ only
   in ahead. Each pair has different signatures.
2. Every tile parameter of `MatMulTile` appears in the signature, and a test fails if a field is
   added to `MatMulTile` without appearing there (exhaustive destructuring).
3. The 1,800-step stdout sha256 of `examples/mnist_mlp.mspec` on `--backend vulkan` is unchanged
   (`b9bb028a…6904`); only kernel ids change.
4. `task ci` exits 0.
