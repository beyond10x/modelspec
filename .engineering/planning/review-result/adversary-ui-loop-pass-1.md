---
format: aep.planning-md/3
id: review-result:adversary-ui-loop-pass-1
kind: review-result
status: active
title: Adversary pass 1 on observatory-ui-iteration-loop
relations:
- reviews: story:observatory-ui-iteration-loop
revision: 1
---
Batch 1 done; starting batch 2 (serial, unloaded) while I read the batch-1 messages.
unit: story:observatory-ui-iteration-loop R-00, worktree wt-3b4d4a17493e, HEAD c1b949b with my 3 untracked files added
verdict: findings 10 (blocker 0, warning 7, note 3); strongest verdict NEEDS-CHANGE
cases: executed 65→83 (node 41→52 run + 3 skipped; Playwright 24 in `pnpm ux` + 7 in my config, 3 skipped), red 6
origin: introduced 10 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 path (the assigned scratch dir)
needs-coordinator: (1) adopt `ui/ux/adversary.config.ts`, a third file beyond the two the brief named, or move its project into `playwright.config.ts`; (2) the first CI run decides ADV1-06 (fonts); (3) the ADV1-05 fix touches `.github/workflows/check.yml`, which is outside the story's scope list

**1. Diff.** `git --no-pager diff --stat` is empty: no tracked file changed. Untracked and mine: `ui/ux/adversary.test.ts`, `ui/ux/adversary.spec.ts`, `ui/ux/adversary.config.ts`. These are test files only; the charter held.

**2. Cases added.** Each was red the first time it ran, alone and unskipped. They are now `skip` with their id.

| case | asserts | now |
|---|---|---|
| adversary.test.ts, 11 cases | ratchet boundaries (counts; steps 5 % symmetric at 20, no band at 0; layout_shift ±0.0099, ±0.0101 and exactly 0.01), missing scenario or key, first landing, baseline-worse with and without a decision, tolerances equal the skill's numbers | green |
| ADV1-01 ×2 (node) | a loosened `tolerances.counts`, or an entry moved from counts to steps, is caught against main | red, skipped |
| ADV1-02 (node) | `update(…, '')` does not take a worse value | red, skipped |
| adversary.spec.ts, 7 cases | the unit's `probe` reports 110vw, a scrollable panel and the state 2 px below the viewport; `AXE_TAGS` report a nameless svg, an invalid autocomplete (wcag21aa) and #888 on #999; control read of the loss | green |
| ADV1-03 ×2 (Playwright) | the screen-only user reads nothing when the loss value has opacity 0, or is covered | red, skipped |
| ADV1-04 (Playwright) | no animation runs under reduced motion at any event | red, skipped |

Red output, verbatim:
```
not ok 11 - ADV1-01 a baseline whose counts tolerance is loosened lets a worse value through, main unchanged
  error: 'reading_load.max 28 against 25 with counts tolerance 0 on main'   operator: 'notDeepStrictEqual'
not ok 12 - ADV1-01 a baseline entry moved from counts to steps lets a worse value through, main unchanged
  error: 'reading_load.max is a count (KEYS), tolerance 0: 32 against 31 is worse'
not ok 13 - ADV1-02 UX_DECISION set to the empty string does not take a worse value into the baseline
  an empty decision id is no decision id   40 !== 25
✘ ADV1-03 … transparent   Error: read loss step 20: 0.0952381 with 0 actions   Received: {"step": "20", "value": "0.0952381"}
✘ ADV1-03 … covered       Error: read loss step 20: 0.0952381 with 0 actions   Received: {"step": "20", "value": "0.0952381"}
✘ ADV1-04   Error: animations per event 1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2; 2.5 s after the end 0   Expected: 0  Received: 2
```

**3. Suite runs** (after the cases existed):
- In the tree, `pnpm test`: `# tests 55 # pass 52 # fail 0 # skipped 3`, exit 0.
- In the tree, `playwright test -c ux/adversary.config.ts`: `3 skipped 7 passed`, exit 0.
- `pnpm ux` was not run in the tree, because the review agents share its `node_modules/.cache/ux`. It ran instead in scratch copies of `ui/`: clean-1, 2 and 3 gave `24 passed`, exit 0, in 76, 68 and 69 s.
- A fresh shallow clone ran `b10x-build-slot task ui-ux`: `24 passed (1.1m)`, exit 0, and `git status` was clean afterwards.

**Hypotheses**

| # | probe (scratch copy, full `pnpm ux`) | observation | verdict |
|---|---|---|---|
| 1a | scrollable parameter panel | gate 1 `scrollable element section[data-ux-region=parameter-panel] … (y)`, plus gates 5 and 4; exit 1 | caught |
| 1b | state 2 px below the viewport, 1280x720 only | gate 2 `run state: not wholly in the viewport (8,701..90,722 of 1280x720)` and completion_miss; only the 6 replays at 1280x720 failed | caught |
| 1c | finished shown one event early | gate 3 `false_complete … event 21 (StepCompleted step 20)` | caught |
| 1d | finished never shown | `replay: the app did not draw the event within 5000 ms`, completion_miss, gate 4 | caught |
| 1e | `tabindex="-1"` on the picker | gate 5 `focusable 0 is not reached by Tab`, 8 replays | caught (the message names no element) |
| 1f | 2 s transition on `d` under reduce | exit 0 (the plant may be inert); a 2 s entrance animation also exits 0 | survives, ADV1-04 |
| 1g | #888 on #999 | gate 5 `axe serious color-contrast` | caught |
| 2a–e | node cases | boundaries pass on both sides; missing scenario gives 11 missing; an extra key gives missing; empty decision is taken | ADV1-01, ADV1-02 |
| 2f | scratch git repo, main carries the baseline, branch worsens it by 1 | with origin/main: exit 1 `the baseline 26 is worse than main's 25 and names no decision`; without the ref: `fatal: invalid object name 'origin/main'`, ratchet `1 passed` exit 0 | ADV1-05 |
| 3 | `bot.ts`/`page.ts` reads; value only in an attribute; value transparent or covered | answers come from `ariaSnapshot` only; attribute-only is caught by gate 4; transparent passes gates 1–5 and only the ratchet objects, as an "improvement"; covered exits 0 | ADV1-03 |
| 4 | 3 runs, `--workers 1`, `--workers 4`, cargo build in parallel, 3 suites in parallel (load average 23–40) | all 7 `round.json` identical, and identical to R-00 | could not break |
| 5 | how the proxy paces; gate-1 count; chart drops every second point | it waits for the event's selector, no sleep; 504 = 503 events + the connecting probe; the dropped points exit 0 (no gate claims the chart's content) | holds; ADV1-07 |
| 6 | dark-only white `.chart` | only the 7 dark replays fail gate 5; R-00's dark shot is dark | caught |
| 7 | UXF assets | UXF-001 and 002 are ratchet keys that work; 003 has no asset, which § 5 allows for an open S3; 004's asset cannot fail; the header test checks fields, id = file and S1–S3 | ADV1-10 |
| 8 | Taskfile, CI | builds modelspec-cli itself; CHROME unset through `task` exits 201, not 2; check.yml sets CHROME; `--workers 4` 141 s, `--workers 1` 381 s (CI default is 2 workers on 4 vCPU, so about 3–4 min, an estimate) | ADV1-09 |
| 9 | public text, lockfile, AGENTS.md claims | no `/home/`, no organization names, no comments; `--frozen-lockfile` exits 0; `-g "L2 layout"` gives 15 passed; `UX_ROUND=R-99` wrote report.md, round.json, shots/ and a trend row; a plain run wrote nothing outside the cache | holds |
| 10 | mutants | table below | ADV1-08 |

**4. Findings** (tree c1b949b plus untracked files)

| id | file:line | verdict / origin | what was measured | what reaches it |
|---|---|---|---|---|
| ADV1-05 | ui/ux/ratchet.spec.ts:47 | NEEDS-CHANGE / introduced | `catch { return null }` treats "ref missing" like "file absent on main"; probe 2f | check.yml uses `actions/checkout` at the default depth 1, so a PR has no `origin/main` and the baseline-worse rule never runs in CI |
| ADV1-01 | ui/ux/core/metrics.ts:153 | NEEDS-CHANGE / introduced | the band comes from the edited file's `tolerances` and each entry's `tolerance`; the main comparison checks values only | any hand edit of baseline.json in a PR |
| ADV1-03 | ui/ux/core/bot.ts:76 | CONFIRMED / introduced | `visibleBox` checks the figure's box and centre; `ariaSnapshot` (:102) also returns text the eye cannot see. Covered variant: `24 passed`, exit 0. Transparent variant: the ratchet says `accuracy reading_load.max: 19 improves on the baseline 25; take it into baseline.json`, and `UX_UPDATE=1` would take it in | skill § 7 "only what is visible"; any styling regression |
| ADV1-04 | ui/ux/layout.spec.ts:188 | CONFIRMED / introduced | `animations()` runs once, after axe and Tab; the entrance-animation variant gives `24 passed`, exit 0 | gate 5 "nothing animates under reduced motion" |
| ADV1-07 | ui/ux/screen-only.spec.ts:109 | CONFIRMED / introduced | the harness waits up to 5 s for the event, then 1 s (`LIVE_BUDGET_MS`). With redraw at 1.5 s per event, accuracy L3 took 45 s instead of 4.8 s and `timeToAct` stayed `{"last-loss": 0, "last-step": 0, "last-accuracy": 0}` | the ratchet key `time_to_act_steps.max` moves only for a lag over about 6 s |
| ADV1-06 | ui/ux/adapter.ts:141 | INFEASIBLE / introduced | forcing DejaVu Sans gives ratchet `linear-regression reading_load.max: 33 is worse than the baseline 31 beyond its tolerance 0`, and the same for two-layer-relu | local `system-ui` is Noto Sans; the CI runner's font I don't know |
| ADV1-08 | ui/ux/layout.spec.ts:201 | CONFIRMED / introduced | the gates run only on a page without defects; mutants c, d, e and f survive `pnpm test` and `pnpm ux` | any later edit that weakens a gate |
| ADV1-02 | ui/ux/core/metrics.ts:215 | CONFIRMED / introduced | `decision !== null` accepts `''`, while the ratchet treats `''` as no decision | `UX_UPDATE=1 UX_DECISION= pnpm ux`; caught locally only once main carries the key |
| ADV1-09 | AGENTS.md:133 | CONFIRMED / introduced | `task: Failed to run task "ui-ux": exit status 2`, then `EXIT=201` | the documented `task ui-ux` |
| ADV1-10 | ui/ux/findings/UXF-004.md:13 | CONFIRMED / introduced | the end-state task reads only the state word, so a fix of "step 20 of 21" and a regression of it both pass | the finding record |

Fixes I would name but did not apply:
- ADV1-05: fetch main in CI, and fail when the ref is missing rather than when the file is absent.
- ADV1-01: compare tolerances and classes with main and `KEYS`, or take them only from `TOLERANCES`.
- ADV1-03: check that the answer's text range is visible, not covered and not transparent.
- ADV1-04: call `animations()` at every probe.
- ADV1-08: adopt my gate self-tests.

**Mutants**

| mutant | killed by |
|---|---|
| (a) glance always 1 | metrics.test `glance`; `pnpm ux` ratchet (`glance_cost.median: 1 improves`) |
| (b) `>=` in the ratchet | metrics.test (2 cases); `pnpm ux` ratchet (`0 is worse than the baseline 0`); adversary.test (10 cases) |
| (c) completion_miss never incremented | survived; it matters only for round.json's count, because `fail()` still gates |
| (d) 3 actions per fix | survived |
| (e) axe tags without wcag21aa | survived the suite; killed by adversary.spec `invalid autocomplete` (`Received array: []`) |
| (f) steps tolerance 50 % | survived the suite; killed by adversary.test `tolerances are the skill numbers` |

**5. Could not break:** the determinism of every gating metric under load and at any worker count; the event pacing; acceptance 1 on a fresh tree; the claim that a plain run writes only to the cache; the dark emulation; the lockfile. I did not attack `task docs-check` (acceptance 6).

**6. Written outside the worktree:** `<wave-root>/ui-loop/adv1-scratch/`. What remains there is logs/, rounds/, taskprobe/ and the *.sh scripts, 1.4 MB. Copies, node_modules and the shallow clone are deleted; its target/ went through `b10x-build-slot cargo clean`. Inside the tree but ignored: `ui/node_modules/.cache/ux/adversary-results/`. The lease is ended.

`git status --short`, my lines: `?? ui/ux/adversary.config.ts`, `?? ui/ux/adversary.spec.ts`, `?? ui/ux/adversary.test.ts`. The other entries are the reviewers': UXF-006 to 011, review-*.md, review-*.png.

```findings
- file: ui/ux/ratchet.spec.ts
  line: 47
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: mainBaseline returns null when origin/main is absent, so the baseline-worse rule never runs in a depth-1 PR checkout such as check.yml's
- file: ui/ux/core/metrics.ts
  line: 153
  category: boundary
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the ratchet takes its tolerance from the baseline under test, so loosening tolerances or switching an entry's class passes against an unchanged main
- file: ui/ux/core/bot.ts
  line: 76
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the screen-only user reads accessible text whose element is transparent or covered, because only the container's box and centre are checked
- file: ui/ux/layout.spec.ts
  line: 188
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: reduced motion is sampled once after the end, so a 2 s animation under prefers-reduced-motion running at every event passes
- file: ui/ux/screen-only.spec.ts
  line: 109
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: time_to_act is 0 by construction because the harness waits up to 6 s for the screen; a 1.5 s render lag per event still measures 0
- file: ui/ux/adapter.ts
  line: 141
  category: property
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: reading_load depends on the resolved font with tolerance 0; DejaVu Sans instead of Noto Sans fails the ratchet with 33 against 31, and the CI font is unverified
- file: ui/ux/layout.spec.ts
  line: 201
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the gates run only on a page without defects, so mutants of completion_miss, MAX_ACTIONS_PER_FIX, AXE_TAGS and the steps tolerance survive pnpm test and pnpm ux
- file: ui/ux/core/metrics.ts
  line: 215
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: update accepts an empty UX_DECISION as a decision and takes a worse value, while the ratchet treats the empty string as no decision
- file: AGENTS.md
  line: 133
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the paragraph says CHROME unset is exit 2, but task ui-ux exits 201
- file: ui/ux/findings/UXF-004.md
  line: 13
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the named regression asset reads only the state word and cannot fail when "step 20 of 21" is fixed or regresses
```
