---
format: aep.planning-md/3
id: story:benchmark-history
kind: story
status: active
title: Every benchmark run in one YAML file, and its history against the opponents
relations:
- decomposes: epic:transformer-vs-pytorch-cuda
- serves: vision:modelspec
scope:
- confidence: cited
  path: benchmarks/runs.yaml
- confidence: cited
  path: crates/modelspec-docs
- confidence: cited
  path: spec/domains/benchmark.yaml
- confidence: cited
  path: website/docs/benchmarks.md
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-23T19:51:50Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-23T19:51:50Z", actor: "human:timo", revision: 3}
---
# Every benchmark run in one YAML file, and its history against the opponents

## Why

The operator, 2026-10-23: "please start documentating the benchmark results (I want to see a yaml
file holding all the runs, and then we want to see how we evolve over time against the opponents".
Today the figures live in seven verification reports, the CHANGELOG and scratch directories that are
deleted after 48 hours; nothing shows ModelSpec's step time per release beside PyTorch's.

## Scope

- An ESS domain `modelspec.benchmark` in `spec/domains/benchmark.yaml` declares a benchmark run:
  date, workload, side (ModelSpec backend or opponent), the ModelSpec version and commit or the
  opponent's version, mode (timed or profiled), batch and hidden width where the workload has them,
  FLOP per step, the µs per step of each repeat and their median, GPU, driver and SM clock, and the
  source (a verification-report id or a CHANGELOG release). Its JSON Schema comes from
  `task schema-generate` like the other domains.
- `benchmarks/runs.yaml` holds every run measured so far, one entry per repeat where the source
  has them, each citing its source.
- A Rust generator (in `crates/modelspec-docs`, beside the other generated pages) reads the file,
  validates it against the schema, and writes `website/docs/benchmarks.md` with one SVG chart per
  workload: µs per step (log scale) over date, one series per side, ModelSpec and the opponents
  together, and a table of the latest run per side. `task docs-check` fails when the page is stale.

## Acceptance

- `benchmarks/runs.yaml` holds the MNIST 784-64-10 runs of ModelSpec CPU 0.1.1, Vulkan 0.4.0, 0.5.0,
  0.6.0 and the 2026-10-23 profile run, PyTorch CPU 1 and 8 threads, PyTorch CUDA eager and
  `torch.compile`, and the 36 points times 3 repeats of `verification-report:mlp-size-sweep-2026-10-23`.
- An entry missing a required field, or with a median that is not the median of its repeats, is
  refused with its line and field by a host-only test.
- `website/docs/benchmarks.md` shows the MNIST chart with every release's ModelSpec Vulkan median
  and the PyTorch CUDA eager line, and the sweep's charts per batch.
- `task ci` exits 0.
