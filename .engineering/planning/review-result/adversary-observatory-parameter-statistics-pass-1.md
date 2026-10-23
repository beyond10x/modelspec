---
format: aep.planning-md/3
id: review-result:adversary-observatory-parameter-statistics-pass-1
kind: review-result
status: active
title: 'Adversary pass 1: Vulkan first and last events empty, chart-name collision with a metric'
relations:
- reviews: story:observatory-parameter-statistics
revision: 1
---
# Adversary pass 1 on story:observatory-parameter-statistics

Commit 6272b74 plus 3 untracked adversary test files. Verdict: one warning, one note, no blocker.
Executed 1417 to 1426 cargo cases and 22 to 26 node cases; 1 red.

Red, verbatim (`node --test … adversary_parameter_stats.test.ts`):

```
not ok 4 - a metric and a parameter norm chart never share a name
  error: |-
    chart names: ["loss","M.w value norm","M.w value norm","M.w gradient norm"]
    3 !== 4
```

`task ci` exited 201 at `[ui] pnpm test` (25 of 26 passed); the cargo lanes before it passed 1418,
failed 0, ignored 130. The two live browser scenarios failed once at load average 33 to 42 and
passed on a rerun with `--test-threads=1` at load 24; read as a timing flake, not a finding.

Attacked without a break: histogram definition, edge bits, last bin, counts against the finite
count, all-NaN and ±MAX tensors (18 fixed and 4000 seeded tensors); L2 norm of 2^20 × f32::MAX;
gradients against finite differences of an independent f64 oracle; dense batch k; sampling with
N > steps, N = 1, resume, resume with zero steps; `--statistics-every 0` and the flag with Vulkan
refused; spec, serde, schema and generated TypeScript agree; UI with empty lists, reconnect resend
and a new run; CPU losses bit-identical to the f4f7310 fixtures; no extra work on a non-sampled
step. Not run: Vulkan device tests; a timing comparison against the base.

```findings
- file: spec/domains/observatory.yaml
  line: 108
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the spec sends every Vulkan step's list empty, including the post-loop event, which the approval record says always carries values, and train.rs:644 emits Vec::new() although the final values are on the host at train.rs:651
- file: ui/src/lib/series.ts
  line: 14
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a metric named "<path> value norm" gives two charts of one name against the doc's "distinct among the charts of a run" (red case ui/src/lib/adversary_parameter_stats.test.ts:117), but no producer emits metrics
```
