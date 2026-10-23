---
format: aep.planning-md/3
id: approval-record:release-0-7-1
kind: approval-record
status: draft
title: Release 0.7.1 approved
relations:
- serves: vision:modelspec
revision: 1
---
# Release 0.7.1 approved

- Requested by b10x-chef-orchestrator on 2026-10-23 under the grant in [the workspace instructions]
  (class C: release): "Continue: PR, fast-forward main, 0.7.1, approved under the grant … Put the
  measured numbers (447.8 GFLOP/s profile, 36/36 sweep, up to 4.26x at batch 8192) in the 0.7.1
  #ai-autodev file per brief.md § Releases; no separate post."
- Point taken: `main` fast-forwarded to 352cc0e, the head of PR 26 (the benchmark wave), after its
  checks were green.
- Scope: the release commit on `main` (workspace and internal dependency versions 0.7.1,
  `CHANGELOG.md` section `[0.7.1] - 2026-10-23`, the 0.7.1 references in `README.md`, `AGENTS.md`,
  `docs/pitch/index.html`, `website/docs/getting-started.md`, `website/docs/index.md`,
  `website/docs/roadmap.md` and `website/docs/status.mdx`, the regenerated
  `website/data/modelspec.terminal.json`), the bot's push of it to `main`, and the bot's tag `0.7.1`
  on that commit.
- Before tagging: `task release-check TAG=0.7.1` exits 0; `check` and `Shared source gates` green on
  the release commit.
