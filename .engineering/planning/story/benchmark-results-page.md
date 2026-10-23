---
format: aep.planning-md/3
id: story:benchmark-results-page
kind: story
status: draft
title: Benchmark results page and pitch-deck slide
relations:
- decomposes: epic:transformer-vs-pytorch-cuda
- depends_on: story:mlp-size-sweep-vs-pytorch-cuda
- depends_on: story:profile-achieved-throughput
revision: 3
---
# Benchmark results page and pitch-deck slide

## Context

`epic:transformer-vs-pytorch-cuda`. The sweep's and the profile's figures are published where
people read them. This story creates the page; `story:transformer-block-benchmark` and
`story:bert-tiny-vs-pytorch-cuda` each add their own table to it later, ordered after this story by
their `depends_on` edges.

## Surface

- New: `website/docs/benchmarks.md` and its sidebar entry.
- Changed: `docs/pitch/index.html`, one new slide.

## Acceptance

- `website/docs/benchmarks.md` exists. It shows the sweep's table from
  `verification-report:mlp-size-sweep-<date>` and one chart of µs per step against FLOP per step
  for the three sides. It also shows the MNIST profile figures from
  `verification-report:mnist-profile-throughput-<date>`. Every figure links to its report.
- The page lists the block benchmark and the BERT-tiny run as "not measured", with no figure.
- `task website` builds the site, exit 0.
- `docs/pitch/index.html` has a new slide, after the MNIST slide, with each side's crossover width at
  batch 1000 and the GFLOP/s of the largest point for the three sides; its notes cite
  `verification-report:mlp-size-sweep-<date>`.
