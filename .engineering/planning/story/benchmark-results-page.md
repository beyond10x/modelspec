---
format: aep.planning-md/3
id: story:benchmark-results-page
kind: story
status: active
title: Benchmark results page and pitch-deck slide
relations:
- decomposes: epic:transformer-vs-pytorch-cuda
- depends_on: story:mlp-size-sweep-vs-pytorch-cuda
- depends_on: story:profile-achieved-throughput
- serves: vision:modelspec
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-26T09:03:19Z", actor: "human:timo", revision: 5}
- {from: "proposed", to: "active", at: "2026-10-26T09:03:19Z", actor: "human:timo", revision: 6}
---
# Benchmark results page and pitch-deck slide

## Context

`epic:transformer-vs-pytorch-cuda`. The sweep's and the profile's figures are published where
people read them. `story:benchmark-history` already created the page: `modelspec-docs generate`
writes `website/docs/benchmarks.md` from `benchmarks/runs.yaml` (every run as a
`modelspec.benchmark.BenchmarkRun`, `spec/domains/benchmark.yaml`), with the MNIST section, the
MLP sweep's three batch sections (µs per step against hidden width and over date) and the table
of every run. This story adds what the epic asks of the page and the deck beyond that: the
FLOP-based view of the sweep, the explicit "not measured" standing of the two transformer
workloads, and the pitch slide. `story:transformer-block-benchmark` and
`story:bert-tiny-vs-pytorch-cuda` each add their own table later, ordered after this story by
their `depends_on` edges (coordinator decision D79, 2026-10-26).

## Surface

- Changed: `crates/modelspec-docs/src/benchmarks.rs` (the generator), `benchmarks/runs.yaml`
  where a stated `flop_per_step` is missing for a sweep run whose report states it, and the
  generated `website/docs/benchmarks.md` and `website/static/img/benchmarks/*.svg`.
- Changed: `docs/pitch/index.html`, one new slide after the MNIST slide.

## Acceptance

- `website/docs/benchmarks.md` shows, for the sweep, one chart of µs per step against FLOP per
  step for the three sides (ModelSpec Vulkan, PyTorch CUDA eager, PyTorch CUDA `torch.compile`),
  from the runs that state `flop_per_step`; and a GFLOP/s column (`flop_per_step` / µs per step
  / 1000) in the sweep's latest-median tables wherever the run states `flop_per_step`, blank
  otherwise. Every figure links to its report, as the page already does.
- The page has a "Not measured" section that lists the transformer-block benchmark and the
  BERT-tiny pre-training comparison by name with no figure, from a list of declared workloads in
  the generator; a declared workload gains runs in `runs.yaml` and leaves that section without
  a generator change.
- The MNIST profile figures of `verification-report:mnist-profile-throughput-<date>` stay on the
  page as they are.
- `modelspec-docs generate --check`, `task docs-check`, `cargo test -p modelspec-docs` and
  `task website` exit 0; a unit test holds the FLOP chart's points and the GFLOP/s values for a
  fixture of runs, and one holds the "Not measured" section.
- `docs/pitch/index.html` has a new slide, after the MNIST slide, with each side's crossover
  width at batch 1000 (the `Crossover` section of `verification-report:mlp-size-sweep-<date>`,
  the rows with `pitch_slide: true`) and the GFLOP/s of the largest point for the three sides;
  its notes cite that report. Slides after it renumber.
