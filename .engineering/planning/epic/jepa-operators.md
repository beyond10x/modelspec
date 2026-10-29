---
format: aep.planning-md/3
id: epic:jepa-operators
kind: epic
status: implemented
title: 'JEPA pretraining operators: stop_gradient and SIGReg on CPU and Vulkan'
summary: The operators a JEPA-style pretraining program needs, for the decider repository as first consumer.
relations:
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T10:45:43Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-28T10:45:43Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-29T20:34:02Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
## Outcome

modelspec can state and train a JEPA-style pretraining program: `stop_gradient` and a `sigreg`
loss operator (the sketched isotropic Gaussian regularizer of LeJEPA, characteristic-function
form), on the CPU interpreter and on Vulkan, held to the CPU within the DESIGN.md § 9 tolerance.

## Context

- First consumer: the private decider repository (operator decision 2026-10-28), which trains a
  decision model on normalized agent sessions with a JEPA objective and describes it in `.mspec`.
  It is a new stakeholder of modelspec beside the transformer-within-3x goal.
- Design: unit U2 of the decider design (§ 5 model, § 8 plan), kept outside this repository.
- Scoped against `origin/main` 780132c (release 0.21.0): none of `stop_gradient`, `sin`, `cos`,
  SIGReg or an EMA parameter class exists; SIGReg cannot be composed in source today.

## Parts

1. `story:jepa-stop-gradient`
2. `story:jepa-sigreg-cpu` (depends on 1)
3. `story:jepa-sigreg-vulkan` (depends on 2)

## Decisions (modelspec coordinator, 2026-10-28)

- Projections are redrawn every step, as dropout draws its mask; this replaces the decider
  design's "fixed projection matrix". The S2 implementor checks the knots, weights and scaling
  against arXiv 2511.08544 and records them in DESIGN.md.
- `stop_gradient` stays in: it is the target branch of the design's § 5.2 and costs half a day.
- `ema_update`, smooth-L1 and concatenation are out of scope until the decider's U3 needs them.
- Runs beside the transformer-within-3x goal, not ahead of it; GPU work waits for the GPU to be
  free of timed runs.

## Out of scope

- The decider's own `.mspec` program and its training run (decider U3).

## Priority

- Modelspec work that blocks the decider goes before all other modelspec work (operator decision
  D88, 2026-10-28). This D88 is the operator's; the D88 in `.engineering/waves/bert-1.md` is a
  different decision, a coordinator's.
- The decider references the local modelspec project. It does not wait for a modelspec release.
- Stories of this epic, all `implemented`, all shipped in release 0.22.0 (CHANGELOG.md § 0.22.0):
  `story:jepa-stop-gradient`, `story:jepa-sigreg-cpu` and `story:jepa-sigreg-vulkan`.
