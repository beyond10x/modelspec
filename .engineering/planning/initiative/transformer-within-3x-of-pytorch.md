---
format: aep.planning-md/3
id: initiative:transformer-within-3x-of-pytorch
kind: initiative
status: draft
title: A transformer within 3× of PyTorch on the GPU
relations:
- serves: vision:modelspec
revision: 1
---
# A transformer within 3× of PyTorch on the GPU

## Outcome

ModelSpec trains a transformer on the GPU through Vulkan at a step time within 3× of PyTorch
eager on the same device, every figure measured on a committed tree and published on the
Benchmarks page, and the comparison against PyTorch closed on time to a held-out loss.

## Where it stands (2026-10-28, release 0.21.0)

| point | ModelSpec Vulkan µs/step | PyTorch eager | torch.compile | source |
|---|---|---|---|---|
| block, batch 8 | 93,868.4 | 3,937.2 (23.8×) | 1,971.9 (47.6×) | `verification-report:transformer-block-benchmark-2026-10-27` |
| block, batch 128 | 641,794.5 | 12,091.1 (53.1×) | 5,138.5 (124.9×) | same |
| MLP sweep, batch 8192, hidden 4096 | 15,635 GFLOP/s | 16,326 | 15,812 | release 0.20.0 |

The sweep shows the matmul kernel is not the gap; the block's small matmuls, attention gradients
and elementwise dispatches are, and the profile unit of step A measures which.

## The order

1. **B**: `story:bert-tiny-vs-pytorch-cuda` closes `epic:transformer-vs-pytorch-cuda`; release 0.22.0.
2. **A**: `epic:transformer-block-within-3x` (drafted beside this initiative): a profile unit
   first, then one unit per measured site, largest first, each measured by the unit 22 bench on
   the committed tree and recorded in `benchmarks/runs.yaml`; done when the block's batch-8
   median is within 3× of PyTorch eager's, measured the same day on the same device and driver.
3. **C**: the decider operators (`stop_gradient`, `sin`/`cos` if missing, a reusable SIGReg loss
   block, `ema_update` only if needed) with CPU/Vulkan agreement tests: one or two units between
   A units, once the operator has answered the decider design's open questions. The model they
   serve trains outside this repository; this repository ships the operators and their tests.
4. **D**: the hardening drafts, at most one per release, by adversary severity, never ahead of a
   ready A unit.

## Rules

- One release after every landed unit. One implementor at a time; an adversary pass on every
  unit; a read-only check of every implementor's claims before the commit.
- Timed device runs are the coordinator's: the GPU alone, clocks never locked, three timed runs
  of at least five seconds, the median. An implementor's smoke run is never a figure.
- Vulkan and SPIR-V stay the only device path. No larger model than BERT-tiny until A is done.
- The short form is the repository skill `goal` (`.agents/skills/goal/SKILL.md`); this record wins
  where they disagree.

## Acceptance

- B: `story:bert-tiny-vs-pytorch-cuda` and `epic:transformer-vs-pytorch-cuda` implemented.
- A: `epic:transformer-block-within-3x` implemented with a `verification-report` whose batch-8
  rows for ModelSpec Vulkan and PyTorch eager, same day, same device and driver, give a ratio
  of at most 3.0.
- C and D: their stories implemented, each under its own release.
