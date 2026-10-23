---
format: aep.planning-md/3
id: approval-record:benchmark-wave-after-m1-h
kind: approval-record
status: draft
title: The wave after m1-h is profile, sweep and results page (orchestrator, 2026-10-23)
relations:
- serves: vision:modelspec
revision: 1
---
# The wave after m1-h is the first three benchmark stories

Decided by the orchestrator on 2026-10-23 under its grant. Timo had asked it to take this
session's decisions (Timo, verbatim: "ask chef to adopt you, chef makes decissions").

## The question

Which stories come after wave m1-h? Two options:
- A: the three benchmark stories of `epic:transformer-vs-pytorch-cuda` that need no new operator.
- B: the BERT operator stories first.

## The decision

b10x-chef-orchestrator, verbatim: "A, approved under the grant. The next wave after m1-h is
profile-achieved-throughput, mlp-size-sweep-vs-pytorch-cuda and benchmark-results-page (drafted at
f25c047 at Timo's request, none needs a new operator). Two rules: Device runs (GPU/CUDA) are
Timo's. Build and test everything that can be checked without a device. When a story reaches the
point where only a device run is left, send me the exact command and what it measures. I put it in
Timo's queue. Do not run it. Start the wave once m1-h is merged and free space is over 30G."

## What it grants

- One wave over `story:profile-achieved-throughput`, `story:mlp-size-sweep-vs-pytorch-cuda` and
  `story:benchmark-results-page`. It starts after wave m1-h merges to `main`, and only with more
  than 30G free on `/`.

## What it does not grant

- Any device run. When a story has only a device run left, the session sends the orchestrator the
  exact command and what it measures. The session does not run it.
