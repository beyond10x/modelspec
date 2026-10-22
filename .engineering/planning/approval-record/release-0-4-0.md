---
format: aep.planning-md/3
id: approval-record:release-0-4-0
kind: approval-record
status: draft
title: Release 0.4.0 approved
relations:
- serves: vision:modelspec
revision: 1
---
# Release 0.4.0 approved

- Approved by b10x-chef-orchestrator on 2026-10-22, class C under the operator's grant (a release
  through the repository's own process), after wave m1-e closed: "Go for release 0.4.0 … with the
  same scope as approval-record:release-0-3-0".
- Scope: the release commit on `release/0.4.0` (workspace version and internal dependency versions
  0.4.0, `CHANGELOG.md` section `[0.4.0] - 2026-10-22`, the 0.4.0 references and the GPU training
  section in `website/docs/getting-started.md` and `website/docs/index.md`, the regenerated
  `website/data/modelspec.terminal.json`), its PR into `main`, and the bot's tag `0.4.0` on that
  commit.
- Conditions: start after PR 18 is merged and its checks on `main` are green; CHANGELOG and docs
  carry only measured numbers with their conditions and do not claim the ADR-007 gate; the claim
  that the binary trains on Vulkan cites the release workflow's build line
  (`.github/workflows/release.yml:76`, `cargo build --locked --release -p modelspec-cli --bin
  modelspec`, default features, which include `vulkan`); a 5xx on a bot write is retried every 10
  minutes and never routed around the bot.
- Before tagging: `task release-check TAG=0.4.0`, `task ci` and `task website` exit 0 in
  `modelspec-release-040`; `check` and `Shared source gates` green on the PR.
