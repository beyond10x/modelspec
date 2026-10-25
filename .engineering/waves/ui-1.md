# Wave ui-1

Serves `vision:modelspec` and the operator's request of 2026-10-25, "make sure we keep the docs
up-to-date, use UI improvement skills": the observatory UI under the `ui-iteration-loop` skill
(hermetic gates on every merge, a baseline that is never made worse, rounds whose change sets are
tasks under one standing story), and the documentation site under the `docs` and
`docs-site-review` skills. Skill `aep:implementing` 0.23.2, wave mode.

## Selection

- `story:observatory-ui-iteration-loop` (new, 2026-10-25): the standing story; R-00 builds the
  levels, each later round is one task under it. Its typed scope is `ui/ux/**`, `ui/package.json`,
  `Taskfile.yml`, `AGENTS.md`, `CHANGELOG.md`; it collides with no open unit (the bert-1 and
  perf-1 lanes touch `crates/`, `DESIGN.md`, `benchmarks/`, `opponents/`), except `CHANGELOG.md`
  and `AGENTS.md`, resolved at the merge as every lane does.
- Dispatch: `aep:implementor` (Opus) per unit, then `aep:adversary` (Opus) per green unit; the
  L7 review agents of a round are two fresh Opus agents the coordinator starts.

## Approval

The operator's words of 2026-10-24, "You have all authority now to make these decisions yourself",
and of 2026-10-25, "use UI improvement skills".

## Units

| # | unit | branch | tree | build dir | scratch | adversary | status |
|---|---|---|---|---|---|---|---|
| 1 | observatory-ui-iteration-loop R-00 (decision D57) | impl/observatory-ui-iteration-loop | wt-3b4d4a17493e | `target/` in the tree | `~/.cache/modelspec-wave-ui-1/ui-loop/scratch/` | | unit c1b949b; adversary pass 1: 10 findings (0 blocker, 7 warnings, 3 notes), L7 review: heuristics 5 pass, 10 partial, 0 fail, UXF-006 to UXF-011 filed; report review: numbers reproduce, T1 to T9 tool defects; fix pass 1 under D59 done (D62): pnpm ux 39 passed, pnpm test 66 passed, R-00 re-measured; task ci exit 0 (2428 passed, 0 failed, 235 ignored) |

## Decisions

| id | decision | reason |
|---|---|---|
| D57 | `story:observatory-ui-iteration-loop` R-00: the loop's files live under `ui/ux/`; L2 and L3 are Playwright specs in TypeScript (the browser side is Vue and TypeScript per the workspace rule of 2026-10-21; the specs drive the browser and are test code of `ui/`), run by `task ui-ux` after `task ui-browser` in `task ci`, against `modelspec observe <fixture> --serve 127.0.0.1:0 --ui ui/dist` with the browser from `CHROME`; the three recorded fixtures are the scenarios (linear regression and two-layer relu the anchors, accuracy the rotating card); gating viewports 1366x768 and 1280x720, measured-only 1440x900, 1920x1080 and 390x844; themes light and dark through `prefers-color-scheme` emulation; no guidance cue, so L0 and gate 6 are dropped; the ratchet lives in the spec and compares `round.json` with `baseline.json`; the code's conventions win where the story's names differ | the skill allows Playwright specs in TypeScript, `ui/` already carries node tests, and a Rust ratchet binary would double the surface for one comparison; the fixtures are the only hermetic inputs the UI has |
| D59 | `story:observatory-ui-iteration-loop` R-00 fix pass 1, from the adversary (ADV1-01 to ADV1-10) and the report reviewer (T1 to T9): the ratchet reads tolerances and key classes only from the code's `TOLERANCES` and `KEYS`, never from the baseline under test (ADV1-01); `origin/main` missing is an error, not "no baseline on main", and `check.yml`'s ui-ux job fetches `main` (ADV1-05; `.github/workflows/check.yml` joins the unit's scope for that one step); the screen-only user reads only text whose own range is in the viewport, uncovered at its centre and not transparent (ADV1-03); reduced-motion is probed at every event (ADV1-04); the end-state task is answerable only in the finished or failed state (T4); `time_to_act` is measured with the stream running at a fixed compressed cadence while the bot reads, lag = events sent minus events drawn (ADV1-07, T3); `layout_shift` is normalised per 100 events, the key renamed `layout_shift.per_100_events` in the adapter and the ratchet, stated in AGENTS.md (T5); the worst-shift moment comes from the worst replay and is shot there (T1); the failed-end replays write results into `round.json` under `failed_end`, outside the ratchet (T2); `round.json` records HEAD, a dirty flag and the `ui/dist` hash (T6); a round rerun deletes only the files it writes, never `shots/review-*` or `review-*.md` (T7); the trend row names its rotating card's columns per scenario (T8); `reading_load` counts the words of text nodes whose box is at least half inside the viewport (T9); the gates run under one injected font stack, `DejaVu Sans` for text and `DejaVu Sans Mono` for code, so the local machine and the CI runner measure the same words (ADV1-06); the adversary's gate self-tests become the `self-test` project of `playwright.config.ts`, run by `pnpm ux` (ADV1-08); `task ui-ux` exits 2 on `CHROME` unset (ADV1-09); UXF-004's asset is the L3 end-state task reading the step count too (ADV1-10); `UX_DECISION=` empty is no decision (ADV1-02); R-00 is re-measured under the fixed definitions, its `report.md`, `round.json` and `shots/` regenerated and `baseline.json` rewritten (no decision id: main carries no baseline), the review files kept; UXF-006 to UXF-011 go to R-01 triage, not this pass | a gate that can be loosened by editing the file it guards, or that passes on a depth-1 checkout, is not a gate; a metric that depends on the run length or the machine's fonts cannot ratchet |
| D62 | R-00 fix pass 1 accepted with its deviations from D59: `time_to_act` is counted in steps with a 250 ms render budget (`RENDER_BUDGET_MS`), which made the three re-measured rounds reproduce byte for byte where raw event counts did not; a caret-edge check beside `elementFromPoint`; fonts verified over CDP `getPlatformFontsForNode`; `task ui-ux` exits 201 through go-task on a missing `CHROME` (2 under `-x`); `check.yml` fetches `main` at depth 1 and installs `fonts-dejavu-core` with the Vulkan packages so the injected font stack exists on the runner; `baseline.json` rewritten without a decision id, `origin/main` holding none | the fix pass report: 3 runs with one `round.json` digest, `pnpm ux` 39 passed, `pnpm test` 66 passed; whether `ubuntu-latest` ships DejaVu is unknown, and the apt line costs one package |
