---
format: aep.planning-md/3
id: review-result:adversary-mlp-size-sweep-pass-1
kind: review-result
status: active
title: 'Adversary pass 1: sweep provenance and GPU-state fairness'
relations:
- reviews: story:mlp-size-sweep-vs-pytorch-cuda
revision: 1
---
# Adversary pass 1 on story:mlp-size-sweep-vs-pytorch-cuda

Tree at 30862e5 on base 4c80e34. Verdict: no failing case; three warnings and three notes on
measurement fairness and provenance. Fixed in 1632acf, `task ci` exit 0 (1465 passed, 0 failed,
131 ignored).

Outcomes:
- 1 fixed: every result line carries `build: {version, commit, dirty}`.
- 2 fixed: the batch and teacher are made once per point before the warm-up (`RunOptions::data`).
  Parameter draws, SPIR-V emission, pipeline creation and buffer setup still run between timed
  runs; their length is not measured.
- 3 fixed: `torch.cuda.synchronize(device)` and `set_device(device)`.
- 4 left: the runner does not expose when the span starts; the bench header states the bias.
- 5 left: upload and read-back sit inside the ModelSpec span (bias under 0.4 %, the adversary's
  estimate).
- 6 recorded: lr 0.001 is approval-record:benchmark-wave-scope D11.

```findings
- file: crates/modelspec-runtime/benches/mlp_sweep.rs
  line: 158
  category: provenance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: result lines named version 0.6.0 and recorded no commit
- file: crates/modelspec-runtime/benches/mlp_sweep.rs
  line: 236
  category: measurement
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: CPU setup repeated before every timed run with the GPU idle, unlike the PyTorch side
- file: opponents/mlp_sweep/mlp_sweep.py
  line: 90
  category: measurement
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: torch.cuda.synchronize was called without the device under test
- file: crates/modelspec-runtime/benches/mlp_sweep.rs
  line: 251
  category: measurement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: span_unix_ms was sampled after the run returned
- file: crates/modelspec-runtime/benches/mlp_sweep.rs
  category: measurement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: upload and read-back are inside the ModelSpec timed span
- file: .engineering/planning/approval-record/benchmark-wave-scope.md
  category: spec-gap
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: lr 0.001 replaced the story's 0.1 without a recorded decision
```
