---
format: aep.planning-md/3
id: story:vulkan-eval-loss-drift-since-unit-20
kind: story
status: active
title: The short device runs reproduce the recorded eval losses again
summary: Find the commit that moved the device eval loss from the unit-20 record and either restore it or move the expectation with a reason.
relations:
- decomposes: epic:m1-vulkan-execution
- serves: vision:modelspec
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T08:47:22Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-29T08:47:22Z", actor: "human:timo", revision: 3}
---
# The short device runs reproduce the recorded eval losses again

## Context

`adversary_gpu_pretrain_cli::device_short_runs_repeat_and_start_where_the_full_run_started` expects
the eval losses unit 20 recorded (9.029848, 8.713591, 8.539593; the full-run fixture
`tests/fixtures/bert_tiny_gpu_pretrain/stdout-lines.txt:3` also holds eval 0 = 9.029848). The
0.22.0 binary and 0782eb1d+52c2fd47 print 9.029767, 8.713636, 8.539629; the step-0 training loss
and the eval accuracy still match. The difference is about 9e-6 relative, far above the 4 ULP of
D141, so the eval loss recipe changed between unit 20 and 0.22.0 without the expectation moving.
Found by the gate triage of perf/bert-step-tenth on 2026-10-29; the test is ignored by default
(device), so no gate saw it. Candidate: the f32 device loss sum of 0.21.0 (D84), not verified.

## Outcome

The commit that moved the eval loss is named, and either the expectation and fixture move to the
new values with that commit and DESIGN.md as the reason, or the eval loss returns to the recorded
values.

## Acceptance

- A bisect between the unit-20 commit and 0.22.0 names the first commit printing 9.029767.
- The device test passes on main under `--include-ignored` with the device inputs set.
