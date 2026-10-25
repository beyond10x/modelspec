---
format: aep.planning-md/3
id: story:shape-mismatch-names-the-disagreeing-operand
kind: story
status: draft
title: A shape mismatch names the operand that disagrees
summary: Every E-TYPE-003 message of an operator with a shape rule names the operand that disagrees with the others, as cross_entropy does; layer_norm is the known case.
relations:
- serves: vision:modelspec
revision: 1
---
# A shape mismatch names the operand that disagrees

## Context

Adversary pass 1 on `story:bert-mlm-cross-entropy` (finding N1): for `logits: f32[5,6]` with `labels` and `weights` of length 6, E-TYPE-003 blamed `labels`. The fix pass made `cross_entropy` name the operand whose row count disagrees with the others. The fix pass found the same defect in `layer_norm`: with `gamma` and `beta` both `[4]` against `x: [6, 5]` the message says `gamma` has type `Tensor<f32, [4]>`, although `gamma` and `beta` agree and `x` is the odd one out. `attention`, `add_rows` and `mse` name no single operand.

## What this story decides

- Every E-TYPE-003 message of an operator with a shape rule names the operand that disagrees with the others, as `cross_entropy` does after its fix pass: when the operands that share a rule agree with each other and the remaining operand disagrees, name that one with its type and the figure it disagrees on; when the operands that share a rule disagree with each other, name both.
- `layer_norm` is the known case; a census of every operator with a shape rule in `types.rs` decides the rest, and each gets a core test of the four cases `cross_entropy_shape_mismatch_names_the_operand_that_disagrees` covers.

## Scope

- `crates/modelspec-core/src/types.rs` (the E-TYPE-003 messages), its unit tests; `website/docs/reference/diagnostics.md` if a wording changes.

## Acceptance

1. `cargo test -p modelspec-core shape_mismatch`: for every operator with a shape rule, the four cases pass and the message names the disagreeing operand.
2. `task ci` exits 0.

## Out of scope

- New error codes.
