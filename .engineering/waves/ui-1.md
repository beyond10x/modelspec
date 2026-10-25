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
| 1 | observatory-ui-iteration-loop R-00 (decision D57) | impl/observatory-ui-iteration-loop | wt-3b4d4a17493e | `target/` in the tree | `~/.cache/modelspec-wave-ui-1/ui-loop/scratch/` | | dispatched: Opus implementor, brief `~/.cache/modelspec-wave-ui-1/ui-loop/brief.md` |

## Decisions

| id | decision | reason |
|---|---|---|
| D57 | `story:observatory-ui-iteration-loop` R-00: the loop's files live under `ui/ux/`; L2 and L3 are Playwright specs in TypeScript (the browser side is Vue and TypeScript per the workspace rule of 2026-10-21; the specs drive the browser and are test code of `ui/`), run by `task ui-ux` after `task ui-browser` in `task ci`, against `modelspec observe <fixture> --serve 127.0.0.1:0 --ui ui/dist` with the browser from `CHROME`; the three recorded fixtures are the scenarios (linear regression and two-layer relu the anchors, accuracy the rotating card); gating viewports 1366x768 and 1280x720, measured-only 1440x900, 1920x1080 and 390x844; themes light and dark through `prefers-color-scheme` emulation; no guidance cue, so L0 and gate 6 are dropped; the ratchet lives in the spec and compares `round.json` with `baseline.json`; the code's conventions win where the story's names differ | the skill allows Playwright specs in TypeScript, `ui/` already carries node tests, and a Rust ratchet binary would double the surface for one comparison; the fixtures are the only hermetic inputs the UI has |
