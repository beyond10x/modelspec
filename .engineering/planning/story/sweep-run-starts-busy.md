---
format: aep.planning-md/3
id: story:sweep-run-starts-busy
kind: story
status: active
title: 'A timed sweep run starts busy: no GPU idle at the start of a run'
relations:
- serves: vision:modelspec
- depends_on: story:fuse-elementwise-chains-and-gradient-sums
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T04:08:55Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-25T04:08:55Z", actor: "human:timo", revision: 3}
---
# A timed sweep run starts busy: no GPU idle at the start of a run

## Context

Timo, 2026-10-24: "modelspec beats python+cuda in the bigger batch benchmark". Coordinator
decision D38 judges the goal on the sweep harness's own back-to-back run. On main 74a9520
(block kernel, chain fusion, layer norm landed) that run measured, at batch 8192, hidden 4096,
8,239.1 µs per step deterministic against 6,579.8 µs for PyTorch eager CUDA (torch
2.14.1+cu130), medians of 3 runs of about 6 s; `runs-fuse-clean/` in the coordinator's scratch,
GPU alone, CPU load 0.54. The block-kernel unit measured the same point at 7,167 to 7,209 µs on
its uncommitted tree (`benchmarks/raw/2026-10-25-blockperf/sweep/r1-final.jsonl`) with the SM
clock steady at 1,680 to 1,710 MHz and utilization 100 % through all three runs
(`r1-final.gpu.csv`).

The clock trace of the main run (`sweep-clocks.csv`, 500 ms samples) shows, in each of the
three timed runs, the SM clock falling to about 855 MHz and the power to about 190 W 0.4 s after
the run starts, then ramping over 2 to 3 s back to 1,695 MHz at the 419 W power cap. Nothing
else in the run differs from the block-kernel unit's. The same dips appear in a run made while
the CPU was loaded by cargo builds (`runs-fuse/`), so the host's load is not the cause. The
driver restarts a process's clock at 780 MHz and raises it by 30 to 45 MHz per 0.5 s (D39), so
an idle GPU of even a few hundred milliseconds inside a timed run costs seconds of reduced
clock. Each dip costs about 13 % of the run; the steady clock of both runs is the same.

What is not known: which commit between the block-kernel unit's tree and 74a9520 introduced the
idle (candidates: the block-kernel fix pass e34687f, chain fusion fca5377, the merge commits),
and what the GPU waits on at the start of a run (a host read, an upload, pipeline creation, a
fence the runner waits on before the first submit of the run, or the harness's warm-up and
calibration runs leaving the GPU idle before the timed window opens).

## Outcome

The dip is diagnosed with a red-capable loop (the clock and power trace, or `--profile`
timestamps, showing the idle) before any hypothesis, bisected to a commit, and explained by one
runtime observation that the mechanism fired. Then the cause is removed so that a timed sweep
run keeps the GPU busy from its first submit to its last wait, and the b8192 h4096 point on
main measures within the run-to-run spread of the block-kernel unit's 7,167 to 7,209 µs, or
better, with the bits of every output unchanged.

## Acceptance

- **idle-reproduced**: `benchmarks/raw/<date>-sweep-idle/` holds a trace (clock, power and
  utilization at 100 ms, or `--profile` dispatch timestamps) of the dip on 74a9520 at b8192
  h4096 and the command that produced it, and one trace of the block-kernel unit's commit or
  binary without it, interleaved in one session.
- **idle-bisected**: `diagnosis.md` there names the commit that introduced the idle, with the
  trace of its parent and of it, and names what the GPU waited on, with the observation (a
  profile line, a fence wait timed on the host, a log of the runner) that shows it.
- **idle-removed**: with the fix, three timed runs at b8192 h4096 show no sample below 1,500
  MHz after the first second of the run, and the median µs per step is within 5 % of 7,188 or
  below it, GPU alone, in the same session as a run of PyTorch eager CUDA; the harness's
  recorded spans and the clock samples are in the raw directory.
- **same-bits**: `train --backend vulkan` stdout sha256 and the checkpoint digest equal 74a9520
  for `examples/mnist_mlp.mspec` and the b8192 h4096 source; `task ci` green.
- **harness-honest**: if the cause is in the harness (`crates/modelspec-runtime/benches/
  mlp_sweep.rs`) rather than the runtime, the fix keeps "host wall time from the first submit to
  the end of the one host wait" as the measured quantity and says in the bench's module
  comment what changed; the PyTorch opponent is not changed.

## Scope

- crates/modelspec-runtime/benches/mlp_sweep.rs
- crates/modelspec-runtime/src/vulkan/runner.rs
- crates/modelspec-runtime/src/vulkan/
- benchmarks/raw/
- benchmarks/runs.yaml
- CHANGELOG.md
- DESIGN.md
