---
format: aep.planning-md/3
id: approval-record:m1-b-start
kind: approval-record
status: draft
title: Land wave m1-a and start wave m1-b without a further go
relations:
- serves: vision:modelspec
revision: 1
---
# Land wave m1-a and start wave m1-b without a further go

- Relayed by b10x-chef-orchestrator on 2026-10-22, under the grant of
  `approval-record:m1-waves-and-device-runs`: fast-forward `main` to the head of PR #13 once all
  three of its checks are green, then start wave m1-b (`story:m1-vulkan-elementwise`) without
  waiting for a further message. GPU runs on the RTX 3090 stay approved (chef's decisions log,
  06:24).
- Done: PR #13's checks passed on `6937dac` (task ci, documentation build, security and privacy);
  `main` fast-forwarded from `a5f2b06` to `6937dac` by the bot. The head moved from `fa65210` to
  `6937dac` because the first CI run lacked the Vulkan headers.
- Wave m1-b runs under `approval-record:m1-tolerance-numerics-class`: per-operator tolerances in
  every GPU test.
- Source: the cross-session messages from b10x-chef-orchestrator.
