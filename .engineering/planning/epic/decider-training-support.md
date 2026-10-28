---
format: aep.planning-md/3
id: epic:decider-training-support
kind: epic
status: active
title: 'Decider training support: train and score the decider model in ModelSpec'
summary: decider.mspec trains on the CPU and through Vulkan on the decider repository's batches, and infer scores held-out decisions; anything blocking the decider goes first.
relations:
- serves: vision:modelspec
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T12:12:29Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-28T12:13:46Z", actor: "human:timo", revision: 3}
---
# Decider training support: train and score the decider model in ModelSpec

## Outcome

`decider.mspec`, the decider's JEPA-style decision model written in ModelSpec (the decider design
`docs/design/decider-v0.1.md` § 5.2 at its stated sizes: H 256, event encoder 4 layers over
256-token events, context encoder 2 layers over k = 32 events, 128 decisions a step, a catalog of
64 templates plus abstain; D-NEXT, one view pair), trains on the CPU and through Vulkan on the
batches the decider repository writes, and `modelspec infer` scores held-out decisions with it.

The file is the gap scout's draft of 2026-10-28, adapted as each story below lands: every
workaround it carries today (selection matmuls for a reshape, a split weight for a concatenation,
smooth-L1 built from `relu`, `sub(0.0, x)` for a sign, an additive `-1e9` bias for absent
candidates, hand-inlined encoder layers, shape-only index tensors shipped as data) is replaced by
the construct the story adds.

## Priority (Timo, 2026-10-28)

Anything in modelspec that blocks the decider repository goes first. The stories are ordered by
what blocks first: S1 is the only one without a workaround, so nothing of the decider trains until
it lands.

## Consumer

The decider repository (beyond10x, private) is the consumer. It owns the extraction of event
tokens and decision records, the tokenizer, the batch files and the full-size runs. ModelSpec owns
the language, the compiler, the runtime and the CLI those runs use.

## Stories, in the order they unblock

1. `story:decider-dense-named-data` (S1, gap row 1): dense `--data` with any number of named `f32`
   and `i32` inputs and targets, on the CPU and on Vulkan, `--eval` included.
2. `story:decider-small-operators` (S2, gap rows 5–10): `reshape`, last-axis `concat`,
   `smooth_l1`, `sqrt`, `abs`, masked `cross_entropy` and negative float literals.
3. `story:decider-infer-on-data` (S3, gap row 3): `infer --data DIR --function NAME --out FILE`.
   Depends on S1.
4. `story:decider-loss-terms-and-rank-probe` (S4, gap row 2): per-term loss records and an
   `effective_rank` probe in the run record.
5. `story:decider-model-function-calls` (S5, gap row 11): calls between model functions, inlined.
6. `story:decider-const-tensors` (S6, gap row 4): `const` tensors with index generators.

The gap list ordered rows 1–13 by what blocks first; S2 is placed second because its operators
remove most of the workarounds and the bulk of the per-step data (`fold_rows` alone is 4.3 MB a
step), not because it is a precondition of S3–S6.

## Also needed: `epic:jepa-operators`

`decider.mspec` also needs `stop_gradient` and SIGReg (`story:jepa-stop-gradient`,
`story:jepa-sigreg-cpu`, `story:jepa-sigreg-vulkan` under `epic:jepa-operators`, planned on branch
commit 08f2026, not yet on main). This epic does not duplicate them and does not depend on them:
every story here is testable on fixtures without them. The closing check below needs both epics.

## Done when (the coordinator's closing check)

- S1–S6 implemented, and `epic:jepa-operators` implemented.
- The decider fixture S1 adds (`crates/modelspec-cli/tests/fixtures/decider_small.mspec`, the
  decider's program at reduced sizes) is rewritten without the workarounds above, with
  `stop_gradient` and `sigreg`, and trains from a dense directory on the CPU and on Vulkan, the
  Vulkan losses within the DESIGN.md § 9 gate 5 tolerance of the CPU's, and `infer --function
  scores` writes its scores; `task ci` green.
- The decider repository reports one full-size run of `decider.mspec` on its own batches on the
  RTX 3090, with the per-term losses and the effective-rank series in the run record, and one
  `infer` scoring pass over its held-out split. That report is the consumer's; this epic records
  it as evidence when it arrives.

## Out of scope

- Gap row 12, a tokenizer for event text (`data tokenizer --text FILE --reserved FILE`, a BPE
  kind): the decider repository's extraction crate owns tokenization and writes token ids, which
  S1 reads as `i32` files.
- Gap row 13, an EMA target (a non-trainable parameter class and a post-step `ema_update`
  effect): needed only for the fallback if SIGReg does not hold on the decider corpus (design
  § 5.1). Planned when a run shows SIGReg failing, not before.
- Full `module` / `layer` (E-SYNTAX-004/005, DESIGN.md § 11 iteration 3): S5 adds inlined calls
  only.
- The design ambiguities A1–A3 of the gap list (the `‖ŝ − s'‖` head term, the reading of
  "target embedding of event t+1", the position-in-session token): questions for the decider
  repository, not modelspec work.
- Calibration of the deferral threshold τ and every evaluation metric of design § 7: computed by
  the decider repository from the scores S3 writes.

## Context

- Gap list: the decider gap scout's `gaps.md` of 2026-10-28 (rows 1–13, run against modelspec
  0.21.0), with its draft `decider.mspec` and `decider-current.mspec` (the same program with the
  two planned operators replaced; `check` exits 0, `train --backend cpu` refuses it with
  E-DATA-001: 15 inputs, 2 targets).
- Decider design § 5.2 (architecture), § 5.3 (training: smooth-L1 latent loss plus λ·SIGReg,
  cross-entropy over candidates with an abstain logit, τ calibrated on held-out sessions), § 5.4
  (where it trains).
