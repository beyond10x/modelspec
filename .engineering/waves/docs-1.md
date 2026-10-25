# Wave docs-1

Serves `vision:modelspec` and the operator's request of 2026-10-25, "make sure we keep the docs
up-to-date": the documentation site, README and AGENTS.md under the `docs` skill, with the
docs-site-review of 2026-10-25 as its Measure step. Skill `aep:implementing` 0.23.2, wave mode.

## Selection

- `story:docs-current-bert-1` (new, 2026-10-25): the hand-written pages, the status data, the
  generators of the reference and benchmark pages, the pitch deck, README and AGENTS.md. Its typed
  scope collides with the open lanes only on `CHANGELOG.md`, `AGENTS.md` and
  `crates/modelspec-docs/src/status.rs` (the rowwise and heldout units add status rows at their
  landing), resolved at the merge as every lane does.
- Dispatch: `aep:implementor` (Opus) for the unit, then `aep:adversary` (Opus): the adversary
  reads the pages against the code and the review's fix list.

## Approval

The operator's words of 2026-10-24, "You have all authority now to make these decisions yourself",
and of 2026-10-25, "make sure we keep the docs up-to-date".

## Units

| # | unit | branch | tree | build dir | scratch | adversary | status |
|---|---|---|---|---|---|---|---|
| 1 | docs-current-bert-1 (decision D58) | impl/docs-current-bert-1 | wt-41996e980b07 | `target/` in the tree | `~/.cache/modelspec-wave-docs-1/current/scratch/` | | dispatched: Opus implementor, brief `~/.cache/modelspec-wave-docs-1/current/brief.md` |

## Decisions

| id | decision | reason |
|---|---|---|
| D58 | `story:docs-current-bert-1`: one unit for every content and own-file defect of the site review plus the CHANGELOG review since 0.7.1, grouped by cause inside the unit; template defects are answered `template` with the docs-system file and routed to the template's owner through the coordinator, never patched in this repository's CSS; generated pages change only through their generator; the fix-list answer file and the 390 px chart shots live in `.engineering/reviews/site-review-2026-10-25/` so the next re-screen has them; the `docs` skill's Stage B is the coordinator's (bot PR, `Documentation validation` and `Documentation site` runs read), Stage D is the re-screen the coordinator runs with `docs-site-review` after the deploy | the review found one cause behind most content findings (pages written for 0.7.1), so one unit is cheaper than one per finding; copying a template fix into one site's CSS would make every other site's fix a second copy |
