---
format: aep.planning-md/3
id: approval-record:release-0-5-0
kind: approval-record
status: draft
title: Release 0.5.0 approved
relations:
- serves: vision:modelspec
revision: 1
---
# Release 0.5.0 approved

- Requested by the operator on 2026-10-23, before the next wave: "commit and cut a new version
  first".
- Scope: the release commit on `release/0.5.0` (workspace version and internal dependency versions
  0.5.0, `CHANGELOG.md` section `[0.5.0] - 2026-10-23`, the 0.5.0 references and the measured step
  time in `website/docs/getting-started.md` and `website/docs/index.md`, the regenerated
  `website/data/modelspec.terminal.json`), its PR into `main`, and the bot's tag `0.5.0` on that
  commit.
- Content: wave m1-f, `story:m1-f-forward-matmul-tiled`; 27.282 µs per step on the MNIST benchmark
  example on an RTX 3090, median of 5 runs of 144,000 steps, mean P0 1957 to 1965 MHz. ADR-007's
  accuracy and final-weight conditions are not claimed.
- Conditions: start after the wave m1-f PR is merged and its checks on `main` are green; a 5xx on
  a bot write is retried and never routed around the bot.
- Before tagging: `task release-check TAG=0.5.0`, `task ci` and `task website` exit 0 in
  `modelspec-release-050`; `check` and `Shared source gates` green on the PR.
