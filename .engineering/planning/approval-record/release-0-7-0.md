---
format: aep.planning-md/3
id: approval-record:release-0-7-0
kind: approval-record
status: draft
title: Release 0.7.0 approved
relations:
- serves: vision:modelspec
revision: 2
---
# Release 0.7.0 approved

- Requested by b10x-chef-orchestrator on 2026-10-23 under the grant in `~/beyond10x/CLAUDE.md`
  (class C: release), quoting Timo at 17:17Z: "dispatch and control: modelspec + patchback". The
  orchestrator: "continue under the grant in ~/beyond10x/CLAUDE.md through merge and release, and
  start the next wave from your store without asking."
- Point taken: `main` at 6fd6c30 after PR 24 (wave m1-h) merged at 2026-10-23T17:35:19Z; `check`,
  `Documentation validation` and `Shared source gates` green on 6fd6c30 in the pull-request runs.
- Scope: the release commit on `release/0.7.0` (workspace and internal dependency versions 0.7.0,
  `CHANGELOG.md` section `[0.7.0] - 2026-10-23`, the 0.7.0 references in `README.md`, `AGENTS.md`,
  `docs/pitch/index.html`, `website/docs/getting-started.md`, `website/docs/guides/write-a-model.md`, `website/docs/index.md`,
  `website/docs/roadmap.md`, `website/docs/status.mdx` and the status rows of
  `crates/modelspec-docs/src/status.rs`, the regenerated `website/data/modelspec.terminal.json` and
  `website/data/status.json`), its PR into `main`, the bot's fast-forward of `main`, and the bot's
  tag `0.7.0` on that commit.
- Before tagging: `task release-check TAG=0.7.0` exits 0; `check` and `Shared source gates` green on
  the release commit.
