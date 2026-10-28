---
format: aep.planning-md/3
id: epic:architectural-review-2026-10-28
kind: epic
status: draft
title: 'Remediate the 2026-10-28 architectural review: 10 findings, 4 concerns'
summary: Vulkan device ownership, CPU/GPU input admission, f32 numerics, resume seed, artifact and output contracts, publication and finalization, observation and server bounds; starts after the JEPA and BERT step-time epics
relations:
- serves: vision:modelspec
- depends_on: epic:jepa-operators
- depends_on: epic:bert-tiny-step-tenth
revision: 1
---
# Remediate the 2026-10-28 architectural review

## Request

The operator, 2026-10-28: file the repository-wide Rust architectural review for later, after the
JEPA operators (epic:jepa-operators) and the BERT step-time work (epic:bert-tiny-step-tenth). It
does not start before both are implemented; the `depends_on` edges say so.

## Source

`docs/reviews/architectural-review-2026-10-28.md`, the review as delivered, with the reviewer's
checkout path replaced by `<checkout>`. Reviewed commit `50ef1cde`. Every finding there carries its
reasoning, a reproduction probe (appendix) and a representative acceptance check. Before a story is
drafted from a finding, re-run its probe on current `main`: the step-time and JEPA work change the
Vulkan runner, the loss and the evaluation paths the review read.

## Findings

| id | priority | finding | evidence in the review |
|---|---|---|---|
| F01 | P1 | Safe Vulkan APIs accept resources from another logical device | Rust probe and Vulkan validation |
| F02 | P1 | Vulkan accepts out-of-range position ids that the CPU rejects | integrated CLI reproduction |
| F03 | P1 | Layer normalization is wrong for constant rows in `f32` | ignored regression rerun |
| F04 | P2 | Resume discards the checkpoint's RNG seed | integrated CLI reproduction |
| F05 | P2 | Finite cross-entropy weights can zero every gradient | ignored regression rerun |
| F06 | P2 | Artifact admission does not establish executable kernel interfaces | mutated artifact round-trip |
| F07 | P2 | Observable outputs lack liveness and materialization rules (F07a DCE, F07b scalar loss storage) | two compiler API reproductions |
| F08 | P2 | Dataset publication mutates output partially before failing | filesystem failure reproduction |
| F09 | P2 | A failed terminal recording still publishes successful completion | injected final-write failure |
| F10 | P2 | Vulkan runner selection discards supplied data | static source finding |
| A01 | concern | Optional observation work runs with no observation sink | static, unbenchmarked |
| A02 | concern | Server lifetime and replay memory are process-scoped and unbounded | lifetime reproduction, static scaling |
| A03 | concern | Recordings lack the identity needed to reproduce a run | static |
| A04 | lower | Fused-chain names accumulate for the life of the compiler process | static |

## Order (the review's own)

1. Device ownership and shared input admission: F01, F02.
2. Numerical recipes with independent `f32` property checks, CPU and GPU together: F03, F05.
3. Continuation state and effective run inputs, then their identity: F04, F10, A03.
4. Compiler and artifact boundary contracts: F06, F07.
5. Dataset publication and terminal finalization under late failure: F08, F09.
6. Observation cost, replay storage and server lifetime: A01, A02; A04 if a long-lived compiler is
   in scope.

## Decomposition

Not drafted yet. When this epic is picked up: one story per group above, each story's acceptance the
review's acceptance check for its findings, written red first; F03 and F05 change DESIGN.md
numerics and need a design revision in the same change.
