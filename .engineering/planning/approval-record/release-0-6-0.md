---
format: aep.planning-md/3
id: approval-record:release-0-6-0
kind: approval-record
status: draft
title: Release 0.6.0 approved
relations:
- serves: vision:modelspec
revision: 1
---
# Release 0.6.0 approved

- Requested by b10x-chef-orchestrator on 2026-10-23 under the grant in `~/beyond10x/CLAUDE.md`
  (class C: release): "Cut a modelspec release at the next point where main is green between
  units. Do not interrupt a running build for it. Approved under the grant in ~/beyond10x/CLAUDE.md
  (class C: release). Record this in your store."
- Basis quoted by the orchestrator: "origin/main is 25 commits past 0.5.0 (2026-10-23), 19 of them
  code (wave m1-g unit 4: checkpoints, Vulkan infer, atomic checkpoint writes), and CHANGELOG
  Unreleased has 6 entries."
- Point taken: `main` at 24e84d0, between units 3 and 4 of wave m1-h, no build running; `check`,
  `Shared source gates`, `Documentation validation` and `Documentation site` green on 24e84d0.
- Scope: the release commit on `release/0.6.0` (workspace and internal dependency versions 0.6.0,
  `CHANGELOG.md` section `[0.6.0] - 2026-10-23`, the 0.6.0 references in
  `website/docs/getting-started.md` and `website/docs/index.md`, the regenerated
  `website/data/modelspec.terminal.json`), its PR into `main`, the bot's fast-forward of `main`, and
  the bot's tag `0.6.0` on that commit.
- Before tagging: `task release-check TAG=0.6.0` exits 0; `check` and `Shared source gates` green on
  the release commit.
