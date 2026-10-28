---
name: goal
description: The current goal of modelspec and the order of work toward it. Use when a session asks what the goal is, what comes next, which unit to pick, or whether a proposed unit serves the goal; and at the start of every coordinating session in this repository.
---

# goal

The governed record is `initiative:transformer-within-3x-of-pytorch` in the planning store
(`aep plan artifact show initiative:transformer-within-3x-of-pytorch`). This file is the short
form; where the two disagree, the store wins and this file is corrected.

## The goal

ModelSpec trains a transformer on the GPU through Vulkan at a step time within 3× of PyTorch
eager on the same device, every figure measured on a committed tree and published on the
Benchmarks page, and the comparison against PyTorch closed on time to a held-out loss.

Where it stands (release 0.21.0, `verification-report:transformer-block-benchmark-2026-10-27`):
the BERT-tiny encoder block at batch 8 takes 93,868 µs per step on ModelSpec Vulkan against
3,937 µs on PyTorch eager (23.8×) and 1,972 µs on torch.compile (47.6×); at batch 128, 53.1×
and 124.9×. The MLP sweep already matches PyTorch at large sizes (15,635 against 16,326 GFLOP/s
at batch 8192, hidden 4096), so the gap is the block's small-matmul, attention-gradient and
elementwise dispatches, not the matmul kernel.

## The order

| step | work | done when |
|---|---|---|
| B | `story:bert-tiny-vs-pytorch-cuda`: BERT-tiny time to a held-out loss, ModelSpec Vulkan against PyTorch CUDA, under the measurement rules of `epic:transformer-vs-pytorch-cuda` | the story is implemented, the epic is implemented, release 0.22.0 is out |
| A | `epic:transformer-block-within-3x`: first a profile unit that names where the 93,868 µs go per site (`train --profile` on `examples/transformer_block.mspec` at batch 8, 32 and 128), then one unit per measured site, largest first; each unit is measured by `cargo bench --bench transformer_block` on the committed tree and its row goes to `benchmarks/runs.yaml` | the block's batch-8 median is within 3× of PyTorch eager's on the same device and driver, measured the same day, both rows published |
| C | the decider operators: `stop_gradient` in the IR, `sin` and `cos` if missing, a reusable SIGReg loss block, `ema_update` only if needed; CPU/Vulkan agreement tests | one or two units; they land between A units once the operator has answered the decider design's open questions |
| D | the hardening drafts (checkpoint reader bounds, validation messages fail the run, observatory server edge cases, the `fast_math` stories): at most one per release, chosen by the severity its adversary gave it | each as a unit; never ahead of an A unit that is ready |

Releases: one after every landed unit (operator rule, 2026-10-25). Sub-agents: one implementor at
a time, an adversary pass on every unit, a Haiku read-only check of an implementor's claims.

## How to pick the next unit

1. `aep plan artifact list --kind story --status active`: an active story is the unit in flight;
   finish it.
2. Otherwise the first row of the table above whose story or epic is not implemented; inside A,
   the site with the largest measured µs that no implemented story has taken.
3. A unit that changes no measured figure (D, docs) runs only when no A unit is ready, or beside
   a release.
4. Timed device runs are the coordinator's: the GPU alone, clocks never locked, three timed runs
   of at least five seconds, the median; an implementor's smoke run is never a figure.

## What the goal is not

Not beating PyTorch on the block; not CUDA or any vendor API (Vulkan and SPIR-V stay the only
device path); not a larger model than BERT-tiny until the block is within 3×; not an observatory
or IR-gallery feature before A is done, unless it measures something A needs.
