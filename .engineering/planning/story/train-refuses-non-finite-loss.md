---
format: aep.planning-md/3
id: story:train-refuses-non-finite-loss
kind: story
status: draft
title: train refuses a run whose loss or parameters stop being finite
relations:
- serves: vision:modelspec
- depends_on: epic:m0-semantic-core
- depends_on: story:observatory-live-event-stream
revision: 2
---
# `train` refuses a run whose loss or parameters stop being finite

## Context

An independent audit of release 0.1.1 (`verification-report:independent-audit-0-1-1`) ran
`linear_regression.mspec` with `sgd(lr = 100.0)`. The loss is `NaN` from step 50 on, every final
parameter is `NaN`, and the process exits 0. The coordinator reproduced it with the 0.1.1 release
binary. A script or CI job that trusts the exit status reads a diverged run as a success.
DESIGN.md § 8 lists the M0 codes for `train` (E-TARGET-001, E-DATA-001 to E-DATA-003) and none for
divergence.

## What it adds

- After each SGD step, the CPU runner checks that the loss and every updated parameter are finite.
- On the first non-finite value the run stops. Stdout keeps every line printed so far and adds the
  `step <k> loss <v>` line of the failing step; stderr carries one diagnostic with a new code
  `E-TRAIN-001`, naming the step, and whether the loss or a parameter (by canonical path) went
  non-finite. The exit status is 1.
- DESIGN.md § 8 lists `E-TRAIN-001` beside the other `train` codes, and its revision line is bumped.
- A run that stays finite prints the same bytes as today.
- `train --serve` (`story:observatory-live-event-stream`): JSON has no `NaN` or `inf`, and the run
  event schema in `spec/` carries a non-finite loss as `null`. The stream's last step event is the
  failing step's, with `loss: null`, and the stream ends after it.

## Scope

- `crates/modelspec-runtime/src/` (the SGD runner), `crates/modelspec-cli/src/train.rs`, a CLI test
  under `crates/modelspec-cli/tests/`, `DESIGN.md` § 8 and the revision line, `CHANGELOG.md`.
- `story:observatory-live-event-stream` (wave observatory-1, running) also edits the runner and
  `train.rs`; this story is scoped against `main` after that wave merges.

## Acceptance

`modelspec train` on a copy of `examples/linear_regression.mspec` with `sgd(lr = 100.0)` exits 1,
prints an `E-TRAIN-001` diagnostic on stderr naming the first non-finite step, and prints no
`param` lines.

1. Divergent run: the `lr = 100.0` fixture exits 1; stderr has exactly one line starting with
   `error[E-TRAIN-001]` (in the existing diagnostic format) that names the step; stdout's last line
   is the `step <k> loss NaN` or `inf` line of that step.
2. Finite run unchanged: `examples/linear_regression.mspec` output is byte-identical to the golden
   `crates/modelspec-cli/tests/golden/` output and exits 0.
3. Parameter case: a fixture whose parameter becomes non-finite on a step where the loss is still
   finite names that parameter's canonical path in the diagnostic.
4. Planted defect: with the check removed, scenario 1 fails (exit 0), recorded in the report.
5. Live stream: the scenario 1 fixture under `train --serve` emits a last step event with
   `loss: null` for the failing step, it validates against the run event schema, and the stream
   then closes.

## Out of scope

- Gradient clipping, learning-rate search or automatic recovery from divergence.
- The Vulkan backend (M1 carries the same check when its runner exists).
