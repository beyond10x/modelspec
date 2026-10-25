# R-00 report review

Reviewer: L7 report reviewer (`references/review-agents.md` § 2), 2026-10-25. Inputs: `report.md`,
`round.json`, `shots/`, `../trend.md`, the three fixtures, `scenarios/*/{truth,tasks}.json`, the
round and metric tool source, `baseline.json`. No questionnaire, no operator notes, no last round.
Nothing paid or live was run.

## Verdict

**Not fit to send as is.** Every number in `round.json` reproduces exactly. Four of the eleven keys
cannot move, or do not mean what their names say, for this round's inputs (§ 1). One number in
`report.md` contradicts the replay that its key is taken from (T1). The report also claims a gate 3
check of which the round keeps no record (T2).

Default action: send R-00 with § 4 prepended to `report.md`. Keep `round.json` and `baseline.json`
as they are, because no value in them is wrong. Fix T1 to T7 before R-01 measures a change. Until
then, triage must not use `layout_shift.per_minute` of linear-regression,
`screen_only.time_to_act_steps.max` or `screen_only.steers` as a target metric.

## § 1 Are the numbers right and meaningful?

How the numbers were recomputed:

- Truth: computed with `jq` from the fixtures. It matches all three `truth.json`, including the
  gradient steps: linear-regression step 400 (3.0406e-5), because step 500 has no Gradient entry;
  two-layer-relu step 100 (0.0591128), because step 200 has none.
- Event counts: 503, 203 and 23.
- Browser-derived values (shift, words, glance levels): these cannot be computed from the JSONL
  alone. They come from a fresh rerun of `pnpm ux` (exit 0, 24 passed); a plain run leaves the
  round dir alone. The aggregations below were then recomputed by hand from that rerun's
  fragments.
- Repeatability: the rerun, the earlier cache run and `round.json` agree on all 33 keys.

| key | round.json (lr / relu / acc) | recomputed | differs | means what it says? |
|---|---|---|---|---|
| layout_shift.per_minute | 0.005405 / 0.013839 / 0.13157 | worst replay (1280x720) sum 0.0045316 / 0.0046823 / 0.0050435, ÷ (503, 203, 23 events / 600) | no | **No.** The same 0.00418 shift at event 0 (UXF-002) dominates every scenario. Run length alone spreads the value 24×. For lr the 0.01 band is larger than the value, so the key cannot improve and fails only above 0.015405 (T5). |
| reading_load.max / .median | 31 / 31 / 25 ; 30 / 30 / 25 | 24 values each: lr and relu {4,10,30×4} at 1280 and {4,10,31×4} at 1366, both themes; acc {4,10,25×4} | no | Partly. It counts only what is above the fold: measured-only 1440x900 gives 40, 1920x1080 43, 390x844 49. Fixing UXF-001 raises it and fails the counts-0 ratchet. The median also counts connecting (4) and started (10). |
| glance_cost.median | 2 / 2 / 2 | 2 by the code; **1 / 1 / 1 by the definition** ("every task the page can answer") | **−1 in each scenario** by the definition | **No.** "How did the run end?" counts as answerable while the page reads `connecting` or `streaming`: 10 of 36 values per scenario, all glance 2 (T4). Without them: lr and relu 16×1, 2×2, 8×3, median 1; acc 16×1, 10×2, median 1. |
| glance_cost.p95 | 3 / 3 / 2 | 3 / 3 / 2 under either reading | no | Yes (UXF-001). |
| actions_per_fix.max / .median | 1 / 1 / 0 ; 0 | per task [0,0,1,0] ×2 viewports; acc [0,0,0,0] ×2 | no | Yes, as redefined: a "fix" is one PageDown to bring an answer into view. The app flags nothing. |
| screen_only.actions / .looks | 1 / 1 / 0 ; 5 / 5 / 4 | looks 1+1+2+1 and 1+1+1+1 | no | Yes. |
| screen_only.steers | 0 / 0 / 0 | 0 | no | Not applicable. There is no cue, and a "steer" (Home from the bottom) only fires when an answer is missing from the page. |
| screen_only.time_to_act_steps.max | 0 / 0 / 0 | 0 | no | **No.** It is 0 by construction (T3). |
| gates | 12 replays, 0 violations, 0 miss, 0 false, 0 axe, 0 wrong | same | no | Gate 1 is the story's weakened form (§ 4). The failed-variant replays are not counted (T2). |
| measured_only reading_load | 25 ×6; 40/43/49 ×2 for lr and relu | same | no | Yes. |

Differences found in `report.md`, none in `round.json`:

- Worst moments, "the largest layout shift, 0.00344": the largest gating shift is 0.00418 at
  1280x720, +0.00074 (+21 %). That 1280x720 replay is also the one `layout_shift.per_minute` is
  taken from. The 0.00344 is the 1366x768 light replay's value, picked because only that replay
  has a shot (T1). UXF-002 does state both values.
- Event 0 accounts for only part of the shift sum. At 1280x720, 0.000351 (lr, 7.7 %), 0.000502
  (relu, 10.7 %) and 0.000863 (acc, 17.1 %) come from later events that the tools do not name. I
  don't know which events.

## § 2 Findings the report misses

| what the recording and shots show | evidence |
|---|---|
| A one-point series draws no mark: at first-step the loss chart and the value-norm chart are empty axes (x −1..1), while the caption reads `step 0: 9.6932` | `shots/linear-regression-1366x768-dark-first-step.jpg`; the contact sheets at first-step, all 3 scenarios |
| The loss and gradient-norm charts look linear, so the curve is flat on the floor for most of the run. The lr loss falls from 9.693 to 5.57e-11 (11 decades); the gradient norm falls from 8.19 to 3.04e-5 and lies flat from step 100 | `linear_regression.run.jsonl` seq 1 and 501; `shots/linear-regression-1366x768-light-first-parameter-gradient-found.jpg`; `linear-regression-contact.jpg` middle-step onward |
| Within one parameter panel the x axes disagree: value norm 0..500, gradient norm 0..400 | same `...-gradient-found.jpg` |
| At the end the default histogram step (latest sampled, 500 or 200) has no gradient, so the panel shows `no gradient at step 500` instead of a gradient histogram | same shot; both anchor fixtures carry Value but no Gradient entry at their last step |
| The second parameter (`TwoLayer.w2`) is not covered by any task, so the bot never measures the two-pages-down cost that UXF-001 names | `scenarios/two-layer-relu/tasks.json` |
| The report gives no explicit "no questionnaire" line among its missing inputs; it is only implied by "no people in R-00" | `report.md` § Not run |

## § 3 Tool defects

| id | file | input that shows it | defect |
|---|---|---|---|
| T1 | `ui/ux/ratchet.spec.ts` (worst-shift pick) and `ui/ux/layout.spec.ts` (shot only at `GATING_VIEWPORTS[0]` light) | any scenario: fragment `layout-*-1280x720-*.json` `worstShift.value` 0.0041806 against the reported 0.00344 | The worst moment is chosen only among replays that have a shot. It is not the worst replay and not the replay the key is taken from. |
| T2 | `ui/ux/layout.spec.ts`, `L2 completion failed` | the accuracy `failedVariant` replays, light and dark | No `writeFragment`. `round.json` `gates.replays` = 12 and the report's gate table leave out the 2 failed-end replays, although bar 3 says they ran. They ran and passed in my rerun. |
| T3 | `ui/ux/screen-only.spec.ts` (live loop) | any scenario | The spec waits for `drawnSelector` (the loss chart's `data-last-step`) before reading. The pacer pushes nothing more until the read finishes, and each lag gets a 1000 ms grace. The loss tasks therefore cannot lag. `last-accuracy` can lag only if it trails the loss chart by more than 1 s. A draw timeout is swallowed (`.catch(() => undefined)`). |
| T4 | `ui/ux/core/bot.ts` `exists`/`glanceNow`; `ui/ux/scenarios/*/tasks.json` `end-state.text` | every scenario at connecting, started, first, middle and last step | The end-state task counts as answerable before the end, because its regex accepts `connecting` and `streaming`. That adds 10 glance-2 values and lifts `glance_cost.median` from 1 to 2. |
| T5 | `ui/ux/core/metrics.ts` `perMinute`, `ratchet`, `update`; `ui/ux/adapter.ts` `EVENTS_PER_MINUTE`, `TOLERANCES.layout_shift` | linear-regression, baseline 0.005405, band 0.01 | The per-minute value on the event clock depends on run length. With an absolute band larger than the value, a fix to 0 is neither reported as improved nor taken in by `update`, and a regression up to 0.015405 passes. For relu, a regression is caught only above 0.023839. The regression asset of UXF-002 therefore guards nothing on lr. |
| T6 | `ui/ux/ratchet.spec.ts` `commit()` | R-00 records `2d83128`, but `git diff 2d83128 c1b949b -- ui/src` shows the `data-ux-region` marks in 4 files that the gates need | The tool records HEAD with no dirty flag and no `ui/dist` hash, so the round names a commit that it did not measure. `pnpm ux` serves whatever `ui/dist` holds; only `task ui-ux` rebuilds it. The `ui/dist` that R-00 used has since been rebuilt (10:41 against the round's 10:37), so it cannot be checked now. |
| T7 | `ui/ux/ratchet.spec.ts` `rmSync(\`${dir}shots\`)` | `UX_ROUND=R-00` rerun after the heuristic review | The rerun wipes `shots/`, including the `shots/review-*.png` files that the review brief tells the heuristic reviewer to put there. 11 such files are in the dir now, untracked. |
| T8 | `ui/ux/core/round.ts` `trendRow` | R-01 with another rotating card | All scenarios go into the row, the rotating card included (the variable is named `anchors`), under a header that is written only once. The new card's values will land under the `accuracy` columns. |
| T9 | `ui/ux/core/page.ts` reading-load walker | 1366x768 first-step: axis label `0.678164` at y≈760 of 768 | The code counts a text node if any of its rectangles touches the viewport, while the definition says "inside". Effect: about 1 word; inferred from the shot, not traced. |

## § 4 Caveats to send with the round

1. Hermetic and synthetic. No people, no questionnaire, no L4, L5 or L6. "Time" is a compressed
   event clock of 600 events per minute, because the fixtures carry no timestamps.
2. Gate 1 here allows the page's own vertical scroll (story acceptance). Skill § 4 allows it only
   for docs sites. Under the skill's gate, linear-regression and two-layer-relu would fail at both
   gating viewports, since the bot's PageDown moves the page. For accuracy it is probably not so,
   judging from the contact sheet, but this is unmeasured.
3. `layout_shift.per_minute` cannot be compared across scenarios. It measures one 0.0042 shift at
   RunStarted divided by run length, and on linear-regression it cannot ratchet (T5).
4. `time_to_act_steps.max` = 0 and `steers` = 0 carry no information this round (T3, § 1).
5. `glance_cost.median` = 2 is an artefact. By its own definition it is 1 (T4).
6. `reading_load` gets worse when UXF-001 is fixed. That fix will need a decision id for the
   ratchet.
7. The screen-only user runs in light theme only. Gate 3's failed end state was checked on the
   rotating card only, and the round files keep no record of it (T2).
8. `commit 2d83128` does not identify the measured code (T6). The measured tree became
   `c1b949b`.
9. The "baseline made worse than main" leg was not exercised: `origin/main` has no
   `ui/ux/baseline.json`.
