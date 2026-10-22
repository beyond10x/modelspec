---
format: aep.planning-md/3
id: approval-record:release-0-3-0
kind: approval-record
status: draft
title: Release 0.3.0 approved
relations:
- serves: vision:modelspec
revision: 1
---
# Release 0.3.0 approved

- Approved by b10x-chef-orchestrator on 2026-10-22, under exception 6 of the operator's grants,
  after it verified `main` `f99308a`: "D: release 0.3.0 from it through your own process (approved
  under the grant in ~/beyond10x/CLAUDE.md), report the tag sha, then stop."
- Scope: the release commit on `release/0.3.0` (workspace version and internal dependency versions
  0.3.0, `CHANGELOG.md` section `[0.3.0] - 2026-10-22`, the 0.3.0 references in
  `website/docs/getting-started.md` and `website/docs/index.md`, the regenerated
  `website/data/modelspec.terminal.json`), its PR into `main`, and the bot's tag `0.3.0` on that
  commit.
- Before tagging: `task release-check TAG=0.3.0`, `task ci` and `task website` exit 0 in
  `modelspec-release-030`; `check` and `Shared source gates` green on the PR.
- Not covered: M1 (Vulkan) work, which waits for the operator because device runs are his.
