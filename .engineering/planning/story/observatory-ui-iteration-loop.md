---
format: aep.planning-md/3
id: story:observatory-ui-iteration-loop
kind: story
status: implemented
title: 'Observatory UI iteration loop: hermetic layout, accessibility and screen-only gates with a ratchet'
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Taskfile.yml
- confidence: cited
  path: ui/package.json
- confidence: cited
  path: ui/pnpm-lock.yaml
- confidence: cited
  path: ui/src/
- confidence: cited
  path: ui/ux/
revision: 11
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T08:54:32Z", actor: "human:timo", revision: 9}
- {from: "proposed", to: "active", at: "2026-10-25T08:54:32Z", actor: "human:timo", revision: 10}
- {from: "active", to: "implemented", at: "2026-10-25T12:01:01Z", actor: "human:timo", revision: 11, decided_on: {"recorded":{"test_result":1,"review_outcome":2}}}
---
# Observatory UI iteration loop: hermetic layout, accessibility and screen-only gates with a ratchet

## Context

The operator asked on 2026-10-25 to "use UI improvement skills" on the observatory UI (`ui/`,
story:observatory-ui-live-loss-chart, story:observatory-parameter-statistics). The
`ui-iteration-loop` skill defines a round loop (Trial, Measure, Triage, Design decision, Build,
Gate) over a baseline that is never made worse, with hard gates on every merge. The UI today is
`ui/src/App.vue` (109 lines) and three components; it follows `prefers-color-scheme` for its two
themes; its browser tests (`crates/modelspec-cli/tests/observatory_ui.rs`) read chart attributes
from a headless Chrome over `modelspec observe <fixture> --serve --ui ui/dist` for the three
recorded fixtures `linear_regression.run.jsonl`, `two_layer_relu.run.jsonl` and
`accuracy.run.jsonl`. Nothing measures layout, accessibility or whether a reader can find a value
on the screen.

## What this story decides

This is the standing story of the loop. Round R-00 builds the hermetic levels the skill runs on
every merge; each later round's change set is one task under this story.

- The adapter and its files live under `ui/ux/`: `adapter.ts` (scenarios, gating and
  measured-only viewports, themes, always-visible elements, regions and moments, completion
  states), `scenarios/<name>/` (the fixture recording the scenario replays, `truth.json` with the
  end state the screen must show, `tasks.json` with the screen-only user's tasks and their
  answers), `baseline.json`, `findings/UXF-nnn.md`, `rounds/R-nn/`.
- L1 is the existing `pnpm test`. L2 (layout contract) and L3 (screen-only user) are Playwright
  specs in TypeScript under `ui/ux/`, run by `task ui-ux` through `pnpm ux` against
  `modelspec observe <fixture> --serve 127.0.0.1:0 --ui ui/dist`, with the browser from `CHROME` as
  the Rust browser tests take it; axe-core through `@axe-core/playwright`. Hermetic: no network,
  no keys, a compressed clock (the replay's step cadence is the recording's own).
- Gates on every merge (skill § 4, without the cue gate 6, because the UI has no guidance cue):
  no scrollable element except the page's own vertical scroll and no horizontal scroll at the
  gating viewports in both themes over every replay frame; the always-visible elements 100 %
  inside the viewport at every state change; `completion_miss = 0` and `false_complete = 0`
  (the finished or failed state of a run is shown when the stream ends and never before); the
  screen-only user completes every task to its truth in at most 2 actions per fix; axe-core 0
  serious or critical violations in both themes and every task doable by keyboard alone.
- The ratchet: `pnpm ux` writes `rounds/R-nn/round.json` and fails on a value worse than
  `baseline.json` beyond its tolerance, on an improvement not taken into the baseline, on a
  missing value, and on a baseline made worse without a decision id. Keys per scenario as the
  skill § 6 names them, minus the cue keys.
- `task ci` runs `task ui-ux` after `task ui-browser`.

## Domain relations

- `modelspec.observatory.UiDirectory` is the served app (`spec/domains/observatory.yaml`); the
  scenarios replay `RunEvent` recordings (`modelspec.observatory.RunEvent`) through
  `modelspec observe`. No new noun: the loop's files are test assets, not entities.

## Scope

- `ui/ux/**` (new): adapter, scenarios, Playwright specs, baseline, findings, rounds.
- `ui/package.json`, `ui/pnpm-lock.yaml`: `playwright`, `@axe-core/playwright` as dev dependencies;
  the `ux` script.
- `Taskfile.yml`: `ui-ux`, in `ci`.
- `AGENTS.md`: an "Observatory UI loop" paragraph under "Observatory UI": the gates, how to run
  them alone, the ratchet rule, the round files.
- `ui/src/**` only for `data-ux-region` marks and accessibility fixes the R-00 gates demand.
- `CHANGELOG.md` one line under Unreleased / Added.

## Acceptance

Every merge runs the hermetic gates; R-00 leaves a baseline and a first measured round.

1. `task ui-ux` (in `task ci`): exit 0 on the tree; the specs run 3 scenarios × 2 gating viewports
   × 2 themes = 12 checked replays and print the per-scenario metric table.
2. `ui/ux/baseline.json` holds every metric key for every scenario; a planted regression
   (a `width: 110vw` on the chart container in a scratch copy) makes `pnpm ux` exit 1 naming
   gate 1 and the viewport.
3. `ui/ux/rounds/R-00/report.md` and `round.json`: the metric table against the bars, the worst
   moments with a screenshot each, the contact sheet of the key moments in both themes.
4. The accessibility gate runs red before the fix on at least one planted violation (an
   `svg[role=img]` without `aria-label` in a scratch copy) and green on the tree.
5. The screen-only user reads, for each scenario, the last loss value, the step of the last
   `StepCompleted`, the gradient norm of the first parameter at the last statistics step (where
   the fixture carries parameters), and the run's end state, each equal to the fixture's truth.
6. `grep -q "ui-ux" Taskfile.yml AGENTS.md` and `task docs-check` exit 0.

## Out of scope

- L4, L5 (keyed, paid), L6 (people) and L8 (real users): operator-only, never in `task check`.
- The round's change set itself (R-01 onward): one task per round under this story.
- rrweb session recording of real users.
