---
format: aep.planning-md/3
id: epic:training-observatory
kind: epic
status: draft
title: 'Training observatory: live training UI over SSE, architecture view and model insights'
summary: Watch a modelspec train run live in a Vue UI over Server-Sent Events, with an architecture graph from Tensor IR and per-step model insights.
relations:
- serves: vision:modelspec
- depends_on: epic:m0-semantic-core
- informed_by: executable-system-specification:modelspec-contracts
revision: 3
---
# Training observatory: a live training UI over SSE

## Outcome

A person runs `modelspec train` and watches it from a browser: loss and metric curves update live
from the training process over Server-Sent Events, the model's architecture is drawn from the
compiled Tensor IR, and per-step insights into the model (parameter and gradient statistics,
activation distributions, attributions for one input) are shown beside it.

## Operator intent

"a training UI connecting via SSE to the training process, visualization, insights to the model,
model architecture visualization, explainability or models" (operator, 2026-10-21).

## In scope

- An event stream from the training process: `modelspec train --serve <addr>` (name to be settled
  in decomposition) exposes `text/event-stream` with one typed event per step and per run lifecycle
  change; replayable from a recorded run file so the UI also works after the run.
- A Rust HTTP server inside the CLI crate (or a new crate) serving the stream and the built UI.
- A browser UI in Vue 3 + TypeScript built with Vite and pnpm (workspace rule, 2026-10-21); event
  and graph types shared with the server are generated from Rust and drift-checked in `task ci`.
- Views: live loss/metric charts; architecture graph from Tensor IR (nodes = operators with shapes,
  edges = tensors, forward and backward distinguished); parameter/gradient histograms and norms
  per step; activation statistics; one explainability method over a selected input (gradient ×
  input saliency first, since the compiler already derives gradients).
- Works for the M0 examples on the CPU backend; nothing in the stream contract is CPU-specific.

## Out of scope

- Multi-user hosting, authentication, remote access beyond localhost.
- Distributed or multi-run comparison dashboards (a later story may add run comparison).
- Any GPU dependency: the observatory must run against `--backend cpu`.

## Acceptance

- `modelspec train --backend cpu examples/linear_regression.mspec` with the serve flag streams one
  event per step; a conformance test reads the stream and checks every event against the generated
  schema and the final loss against the CLI's printed loss.
- The UI renders the linear-regression and two-layer-relu runs live and from a recording; a
  headless browser test checks the loss chart's last point equals the stream's last loss.
- The architecture view of two-layer-relu shows every Tensor IR operator with its shape.
- Saliency for one input of two-layer-relu matches a finite-difference check within 1e-2.

## Specification

The nouns are modelled in `spec/domains/observatory.yaml` (RunEvent, ModelGraph, Insight; a
run's stream is its RunEvents in `sequence` order) and `spec/domains/training.yaml` (TrainingRun
owns many RunEvent). `ess specify validate --path .` in `spec/`: `modelspec v1 — 7 file(s), valid`.
Open questions are the 10 `UNMAPPED:` markers in `observatory.yaml`; each has one owning story,
which settles it first.
