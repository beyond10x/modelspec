---
format: aep.planning-md/3
id: story:step-sites-beat-pytorch
kind: story
status: implemented
title: 'The b8192 h4096 step beats PyTorch eager: the sites outside the two big matmuls'
summary: Pairs every dispatch site of the b8192 h4096 step with PyTorch's kernels in one session, changes the group with the largest gap while keeping the bits, and judges the goal on the back-to-back sweep run.
relations:
- serves: vision:modelspec
- depends_on: story:sweep-run-starts-busy
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T07:30:13Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-25T07:30:13Z", actor: "human:timo", revision: 3}
- {from: "active", to: "implemented", at: "2026-10-25T14:48:08Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1,"review_outcome":1}}}
---
# The b8192 h4096 step beats PyTorch eager: the sites outside the two big matmuls

## Context

Timo, 2026-10-24: "modelspec beats python+cuda in the bigger batch benchmark". Coordinator
decision D38 judges the goal on the sweep harness's own back-to-back run (modelspec then PyTorch
eager CUDA, one thermal session). After `story:sweep-run-starts-busy` the harness no longer idles
the GPU between runs, and the figure is modelspec 7,084.6 µs per step against PyTorch eager
6,420.2 µs (ratio 1.103; `benchmarks/raw/2026-10-25-sweep-idle/final/`). The gap is kernel time.

`benchmarks/raw/2026-10-25-blockperf/diagnosis.md` profiles the step per matmul site on the
final block-kernel tree: site 0 (`[8192,784] x [784,4096]`) 2,600.0 µs at 20,236 GFLOP/s, site 14
(`[784,8192]ᵀ x [8192,4096]`) 2,766.6 µs at 19,018 GFLOP/s, sites 2, 9 and 10 (the `N = 10`
matmuls) 267.5 + 160.1 + 316.3 µs, steady step 6,968.7 µs. The two big matmuls are 5,366.6 µs
of it, which is 77 %; the three small-N matmuls are 744 µs; the remaining 858 µs is every other
dispatch of the step (the elementwise chains, the loss, the reductions, the parameter updates).
Nothing yet says how PyTorch splits its 6,420 µs across the same work, so nothing says which of
the three groups carries the 664 µs gap.

## What this story decides

- One session on the RTX 3090, GPU alone, clocks never locked: `train --backend vulkan
  --profile` of the b8192 h4096 source (the sweep's source, as
  `benchmarks/raw/2026-10-25-blockperf/final/` recorded it) and `torch.profiler` over the same
  step of `opponents/mlp_sweep/mlp_sweep.py` in eager mode (the opponent script gains a
  `--profile` flag that prints per-kernel CUDA time for the timed steps; the model and the
  timed loop are unchanged). The two profiles are paired site by site in
  `benchmarks/raw/<date>-step-sites/pairing.md`: for each modelspec site its dispatch and mean
  µs, the PyTorch kernel(s) that do the same work and their mean µs, and the difference.
- The site group with the largest summed difference gets the change; the change keeps the bits
  (the CPU oracle, `samebits` as the block-kernel and fusion units recorded it) and the
  deterministic reduction orders of DESIGN.md § 4 and ADR-009. Candidates, each a known lead:
  the small-N matmuls at sites 2, 9 and 10 (DESIGN.md records the 4-tile unrolled `Fma` chain
  regressing at `n = 10`; a narrow-N kernel is the lead), the parameter-update and loss
  dispatches (one dispatch per parameter against PyTorch's fused optimizer step), and any
  elementwise dispatch that a chain did not absorb.
- Miss rule: if the pairing shows that no group's change can close 664 µs with the bits kept,
  the story closes with the pairing and a verification report naming the figure each candidate
  reached, and the goal is re-stated in `.engineering/waves/perf-1.md` with that evidence.

## Order on shared files

Perf lane after `story:sweep-run-starts-busy`; shares `crates/modelspec-runtime/src/vulkan/`
with nothing in flight on the BERT chain (CPU-only units).

## Domain relations

None new. A profile site is the existing `--profile` output; the pairing is a raw record.

## Scope

- `opponents/mlp_sweep/mlp_sweep.py` (`--profile`, per-kernel CUDA time; the timed loop unchanged).
- `crates/modelspec-runtime/src/vulkan/` (the kernel or scheduler change the pairing selects),
  its shaders, `crates/modelspec-runtime/tests/` (bits and plan tests for the change).
- `benchmarks/raw/<date>-step-sites/` (profiles, pairing, same-bits, the sweep records),
  `benchmarks/runs.yaml` (the back-to-back sweep entry), `CHANGELOG.md` one line, DESIGN.md
  where a kernel rule changes and its revision line.

## Acceptance

The sweep harness's own back-to-back run at b8192 h4096, deterministic mode, GPU alone, puts
modelspec at or under PyTorch eager CUDA, with the same bits as main.

Scenarios, each run by `task ci` through the command shown unless it needs the device:

1. `benchmarks/raw/<date>-step-sites/pairing.md` pairs every modelspec dispatch site of the step
   with PyTorch kernels and sums the difference per group (big matmuls, small-N matmuls, the
   rest); the group changed is the one with the largest summed difference, stated there.
2. `cargo test -p modelspec-runtime <the change's plan test>`: the plan for the b8192 h4096
   source on the RTX 3090 profile selects the new kernel or dispatch shape at the sites the
   pairing named, and no other site changes.
3. Device, ignored, recorded in the raw directory: `train --backend vulkan` stdout sha256 and
   checkpoint digest on `examples/mnist_mlp.mspec` and the b8192 h4096 source equal main's.
4. `cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep -- --bench --point
   8192:4096 --min-seconds 6` then `opponents/mlp_sweep/mlp_sweep.py --point 8192:4096 --modes
   eager` in one session: the modelspec median of three timed runs is at or under the PyTorch
   eager median; no clock sample below 1,500 MHz after the first second of a run; recorded in
   `benchmarks/runs.yaml`.
5. `task ci` exits 0.
6. Miss rule as above: the pairing and the verification report close the story without the
   kernel change, and say which candidate reached what.

## Out of scope

- `fast_math` and cooperative matrices (`story:fast-math-cooperative-matrix`): the goal is
  deterministic mode.
- The MNIST-sized k11 site (`story:weight-grad-matmul-structural`).
- Changing the PyTorch model, its dtype, or enabling TF32: the opponent stays as it is.
