---
format: aep.planning-md/3
id: approval-record:m1-f-wave
kind: approval-record
status: draft
title: 'Wave m1-f approved: the two matmuls, with measured thresholds'
relations:
- serves: vision:modelspec
revision: 1
---
# Wave m1-f approved: the two matmuls, with measured thresholds

- Relayed by b10x-chef-orchestrator on 2026-10-22 after release 0.4.0 was verified: wave m1-f is
  approved (class C, a wave) with the thresholds and shared conditions as proposed, including the
  miss rule. One implementor at a time, one build at a time through `b10x-build-slot`; pause when
  the chef orchestrator says so for disk.
- Units: `story:m1-f-forward-matmul-tiled` (the `k0` site at most 9.0 µs, median of 3 profiled
  runs), then `story:m1-f-weight-grad-matmul-tiled` (the `k11` site at most 4.0 µs, and the closing
  re-time at most 31.4 µs per step at a mean P0 clock within 1935 to 1965 MHz).
- Shared conditions: results within the per-operator tolerance of
  `approval-record:m1-tolerance-numerics-class` against the CPU oracle; the same stdout sha256 over
  5 runs; `task ci` green; the non-matmul dispatches unchanged.
- Miss rule: if the closing re-time misses 31.4 µs per step, the wave closes with the measured
  number, the gap becomes a new story, and nothing claims the ADR-007 gate.
- Commits this grants: one per unit, the merges into `wave/m1-f`, the closing store commit and the
  fast-forward of `main`, all through the bot; no tag or release.
- Device runs on the RTX 3090 stay under `approval-record:m1-waves-and-device-runs`.
- Source: the cross-session messages from b10x-chef-orchestrator.
