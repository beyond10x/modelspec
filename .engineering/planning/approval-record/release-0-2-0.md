---
format: aep.planning-md/3
id: approval-record:release-0-2-0
kind: approval-record
status: draft
title: Release 0.2.0 approved
relations:
- serves: vision:modelspec
revision: 1
---
# Release 0.2.0 approved

- Approved by b10x-chef-orchestrator on 2026-10-22 at 04:0x, under exception 6 of the operator's
  grants: "Go: df / 85 GB at 04:0x (your df -m). Run task ci in modelspec-release-020, then the bot
  PR, the merge and tag 0.2.0 through your release process (0.2.0 approved earlier in your plan).
  One cargo run at a time."
- The resume message of 2026-10-22 approved continuing wave observatory-1 "through merge and a
  release".
- Scope: the release commit on `release/0.2.0` (workspace version and internal dependency versions
  0.2.0, `CHANGELOG.md` section `[0.2.0] - 2026-10-22`, the 0.2.0 references in
  `website/docs/getting-started.md` and `website/docs/index.md`, the regenerated
  `website/data/modelspec.terminal.json`), its PR into `main`, and the bot's tag `0.2.0` on that
  commit.
- Before tagging: `task release-check TAG=0.2.0`, `task ci` and `task website` exit 0 in
  `modelspec-release-020`; `check` and `Shared source gates` green on the PR.
