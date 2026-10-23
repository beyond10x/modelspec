---
format: aep.planning-md/3
id: approval-record:m1-h-wave
kind: approval-record
status: draft
title: 'Wave m1-h approved: five stories in one integration pull request'
relations:
- serves: vision:modelspec
revision: 1
---
# Wave m1-h approved: five stories in one integration pull request

- Approved by `b10x-chef-orchestrator` on 2026-10-23 under the orchestrator's grant, quoting the
  operator: "keep modelspec running". The orchestrator's words: "continue with modelspec's next open
  item from its AEP store (the wave after m1-g). Approved under the grant in ~/beyond10x/CLAUDE.md.
  Disk is at 60G free; run one cargo build at a time and stop builds under 25G free."
- No story was active after wave m1-g, so the coordinator selected these drafts and moves them
  draft, proposed, active:
  1. `story:train-vulkan-refuses-non-finite-loss` (filed by wave m1-g, unit 2).
  2. `story:plan-verifier-bounded-on-untrusted-artifact` (filed by wave m1-g, unit 3).
  3. `story:m1-f-weight-grad-matmul-columns` (filed by wave m1-g, unit 1).
  4. `story:observatory-ui-live-loss-chart` (its two dependencies are implemented).
  5. `story:observatory-parameter-statistics` (depends on unit 4; starts after unit 4 merges).
- Not selected: the `bert-*` stories, whose first wave (`aep plan artifact waves`) is
  `bert-corpus-splits`, waiting on the operator's corpus and licence decision, and
  `bert-embedding-lookup`, which depends on the draft `story:observatory-saliency-gradient-x-input`;
  the `ir-gallery-*` stories, left for a later wave.
- Units 1 to 4 run in parallel, each in its own tree; every cargo build goes through
  `b10x-build-slot` with one slot and a 25G floor. Merges into `wave/m1-h` in the order above.
- Commits this grants: one per unit and its fixes, the merges into `wave/m1-h`, the store commits and
  the fast-forward of `main` once the one pull request is green, all through the bot. No tag or
  release.
- Device runs on the RTX 3090 stay under `approval-record:m1-waves-and-device-runs`; a timed run
  gets the GPU alone.
