---
format: aep.planning-md/3
id: approval-record:parameter-statistics-settlements
kind: approval-record
status: draft
title: 'Parameter statistics: 32-bin histograms, L2 norm, every N steps (orchestrator, 2026-10-23)'
relations:
- serves: vision:modelspec
revision: 1
---
# Parameter statistics: 32-bin histograms, L2 norm, sampled every N steps

Decided by the orchestrator on 2026-10-23 under its grant (Timo, verbatim: "ask chef to adopt you,
chef makes decissions"). It settles the three `UNMAPPED:` markers of `spec/domains/observatory.yaml`
that `story:observatory-parameter-statistics` reserves for the operator.

## The questions

1. Histogram. A: 32 bins, edges evenly spaced between the tensor's finite min and max, recomputed
   each time statistics are taken. B: fixed edges set per run.
2. Norm. A: L2. B: L2 and max.
3. Sampling. A: `--statistics-every N`, default 100, the final post-loop event always carries
   values (every step on MNIST's 144,000 steps would put about 46M values into one recording).
   B: every step.

## The decision

b10x-chef-orchestrator, verbatim: "Decided now, do not wait for 18:00: Q1 A, Q2 A, Q3 A. Record
this as the approval-record for unit 5 and replace the UNMAPPED lines in observatory.yaml with the
declarations. Additions for Q1 so the spec is total: when finite min == max, emit one bin holding
every value; count NaN and +/-inf separately beside the bins, never inside them. For Q3, the first
step also carries values, so a run shorter than N still has two points. Reason: A is cheaper on all
three (your numbers: 46M values per MNIST recording at every step) and none of them is on the
never-list."
