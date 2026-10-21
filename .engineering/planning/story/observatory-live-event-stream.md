---
format: aep.planning-md/3
id: story:observatory-live-event-stream
kind: story
status: active
title: Stream a CPU training run as typed RunEvents over Server-Sent Events
summary: modelspec train --backend cpu with a serve flag streams one schema-valid RunEvent per step and per lifecycle change; settles the live transport and run lifecycle markers.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
scope:
- confidence: inferred
  path: .github/workflows/check.yml
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: inferred
  path: Cargo.lock
- confidence: cited
  path: Cargo.toml
- confidence: cited
  path: Taskfile.yml
- confidence: cited
  path: crates/modelspec-cli/Cargo.toml
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/
- confidence: cited
  path: crates/modelspec-observatory/
- confidence: cited
  path: crates/modelspec-observatory/schema/
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: inferred
  path: spec/domains/training.yaml
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/docs/reference/crates.md
revision: 22
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T20:16:24Z", actor: "human:timo", revision: 21, decided_on: {"recorded":{"review_outcome":3}}}
- {from: "proposed", to: "active", at: "2026-10-21T20:16:24Z", actor: "human:timo", revision: 22, decided_on: {"recorded":{"review_outcome":3}}}
---
# Stream a CPU training run as typed RunEvents over Server-Sent Events

## Context

The first tracer bullet of `epic:training-observatory`: `modelspec train --backend cpu` with a serve
flag exposes the run as `text/event-stream`, one typed event per step and per run lifecycle change
(epic, In scope, first bullet; first Acceptance bullet). No UI yet: a conformance test is the
consumer. Owns the stream contract that every later observatory story extends; nothing in it is
CPU-specific (epic, In scope, last bullet).

Today `runner::run` returns a finished `Run` with all losses at once
(`crates/modelspec-runtime/src/train/runner.rs`), and `modelspec train` prints after the run
(`crates/modelspec-cli/src/train.rs`). This story adds a per-step observer to the runner so events
leave while the run is going, without changing what stdout prints.

## Open questions settled first

Two `UNMAPPED:` markers of `spec/domains/observatory.yaml` stand in front of this story. They are
named by where they sit and by a quoted fragment of their text, never by line number. Each is
settled before any code, in `spec/domains/observatory.yaml`, with `ess specify validate --path .`
in `spec/` passing afterwards. A settlement names its source (the operator, DESIGN.md or another
ESS document); where none answers, the story stops at that step and reports the question instead
of choosing.

1. Domain-level marker `UNMAPPED (live transport)`, text "the transport of the live stream (the
   `train` serve flag, its address": the serve flag name and its address form, the HTTP path of the
   stream, the SSE framing (`event:` name, `id:` as `sequence`, `data:` as the JSON of one
   `RunEvent`), whether the process keeps serving after the run ends, and that it binds to
   localhost only (epic, Out of scope: no remote access). The sibling marker
   `UNMAPPED (recorded file)` belongs to `story:observatory-run-recording-replay`.
2. Marker on `modelspec.observatory.RunEventBody`, text "RunStarted, RunFinished and RunFailed are
   lifecycle changes of the run": which states a run passes through and what `RunStarted`,
   `RunFinished` and `RunFailed` carry (a failed run carries its diagnostic code from DESIGN.md
   § 8.1?). Today they are unit variants and `TrainingRun` has the single state `Recorded`. Also
   settled here: whether a `StepCompleted` is sent for each of the `steps + 1` losses the runner
   records (loss after `k` updates, `k` in `0..=steps`, `runner.rs` module doc), which is what makes
   the last event equal the CLI line `step <steps> loss <v>`.

## Domain relations

- `TrainingRun` owns many `RunEvent`: `spec/domains/training.yaml`, entity
  `modelspec.training.TrainingRun`, relation `events` (owns, cardinality many, via
  `training_run_id`). A stream is the events of one run in `sequence` order
  (`spec/domains/observatory.yaml`, entity `modelspec.observatory.RunEvent`, field `sequence`).
- `TrainingRun` references one `CompiledArtifact`: `spec/domains/training.yaml`, relation
  `artifact`. M0 writes no `CompiledArtifact` and no `TrainingRun` record
  (`story:m0-cpu-sgd-training`, Out of scope). This story mints a `TrainingRunId` per invocation for
  the events to carry and writes no `TrainingRun` record; whether `RunStarted` carries the record
  fields is part of the lifecycle settlement above.

## Scope

Derived 2026-10-21 by `story-scoper`. Every line is **cited** (read from the story or the tree) or
**inferred** (a reading that could be wrong).

- **Primary surface:** `crates/modelspec-observatory/` (new crate: serde types of `RunEvent` and what it contains, the SSE server, the JSON Schema check) — cited
- **Files:** `crates/modelspec-runtime/src/train/runner.rs` (per-step observer; `run` at line 53 keeps its result) — cited
- **Files:** `crates/modelspec-cli/src/train.rs` (serve flag; `train::Args` is declared here, `runner::run` called at line 85) — cited
- **Files:** `crates/modelspec-cli/Cargo.toml` (gains the observatory crate dependency) — cited
- **Files:** `Cargo.toml` (workspace `members` and `[workspace.dependencies]`) — cited
- **Files:** `Cargo.lock` (new crate and new third-party dependencies: serde, an HTTP server, a JSON Schema validator) — inferred, follows from the new crate
- **Files:** `Taskfile.yml` (`ci` gains `ess specify validate` in `spec/` and the schema `--check`) — cited
- **Files:** `spec/domains/observatory.yaml` (the markers `UNMAPPED (live transport)` and the one on `RunEventBody`) — cited
- **Files:** `crates/modelspec-observatory/schema/` (generated `RunEvent` JSON Schema, outside `spec/`) — cited (the story says "for example")
- **Files:** `crates/modelspec-observatory/tests/` or `crates/modelspec-cli/tests/` (scenarios 2-5) — cited
- **Symbols:** `modelspec_runtime::train::runner::{run, Run, RunError}`, `modelspec.observatory.RunEvent`, `RunEventBody`, `StepCompleted`, `TrainingRunId` — cited
- **Also likely:** `spec/domains/training.yaml` (`TrainingRun.lifecycle` has `states: [Recorded]` at line 156; the lifecycle settlement may add states) — inferred
- **Also likely:** `.github/workflows/check.yml` (installs only go-task and Rust; a `task ci` that runs `ess` needs an `ess` install step pinned to `spec/ess-inputs.yaml`) — inferred
- **Also likely:** `crates/modelspec-docs/src/` stays unchanged: `cli.md` comes from clap (`cli.rs`) and `crates.md` from `cargo metadata` (`crates.rs:1`); only their output pages change — inferred
- **Documents:** `website/docs/reference/cli.md`, `website/docs/reference/crates.md` (via `task docs-generate`), `AGENTS.md` (State: "four crates"), `CHANGELOG.md` (one line under `## [Unreleased]`) — cited
- **Confidence:** high, because the story names every code file and the tree confirms each one. Medium for the CI workflow and `training.yaml`, which the story does not name.
- **Would collide with:** any unit touching the `modelspec-runtime` training loop (`train/runner.rs`), the `train` CLI surface (`modelspec-cli/src/train.rs`), workspace manifests (`Cargo.toml`, `Cargo.lock`), `Taskfile.yml` `ci`, `spec/domains/observatory.yaml` or `spec/domains/training.yaml`, the generated reference pages `cli.md` and `crates.md`, `AGENTS.md` State, or `CHANGELOG.md` `[Unreleased]`
- **Safety fact:** `runner::run` has exactly two callers, `crates/modelspec-cli/src/train.rs:85` and `crates/modelspec-cli/tests/train_linear_adversary.rs:550`, so an observer added beside an unchanged `run` signature changes no existing caller — step 2 (`git grep -n 'runner::'`), unproven
- **Not touched:** `DESIGN.md`. § 7 is the starting sketch of three crates and already omits `modelspec-docs`, so the new crate does not edit it; this keeps `DESIGN.md` and its revision line with `story:observatory-input-adjoint` alone in the wave — coordinator decision, 2026-10-21

## Acceptance

Conformance scenarios, each a named test run by `task ci` except scenario 1, which is the command
shown:

1. `spec/domains/observatory.yaml` no longer contains the marker `UNMAPPED (live transport)` nor the
   marker on `RunEventBody` with "RunStarted, RunFinished and RunFailed are lifecycle changes"; typed
   declarations that cite their source stand in their place, `UNMAPPED (recorded file)` is still
   there, and `ess specify validate --path .` in `spec/` exits 0.
2. `observatory.stream.linear-regression-events-conform`: `modelspec train --backend cpu
   examples/linear_regression.mspec` with the serve flag; a client reads the stream to its end and
   every event validates against the JSON Schema `ess generate --kind schema` produces for
   `modelspec.observatory.RunEvent`; `sequence` runs 0, 1, 2, ... without a gap; all events carry one
   `training_run_id`; every `StepCompleted` has empty `metrics` and `parameters`.
3. `observatory.stream.final-loss-equals-cli`: the loss of the last `StepCompleted` equals, as `f32`
   shortest round-trip text, the CLI line `step 500 loss <v>` of the same invocation, and stdout is
   byte-identical to a run without the serve flag.
4. `observatory.stream.lifecycle-order`: the first event is `RunStarted` and the last `RunFinished`;
   a run the data check refuses (an `E-DATA-003` source, as in `runner.rs` test
   `a_data_refusal_stops_the_run_before_any_step`) ends its stream with `RunFailed` as settled.
5. `observatory.stream.schema-current`: `ess generate --kind schema --check` against the committed
   schema passes in `task ci`.

## Out of scope

- Recording to a file and replay (`story:observatory-run-recording-replay`).
- The browser UI and generated TypeScript types (`story:observatory-ui-live-loss-chart`).
- Filling `StepCompleted.metrics`: no M0 producer computes a metric besides the loss;
  `story:observatory-ui-live-loss-chart` covers metric charts with an `accuracy` fixture recording
  it writes.
- Filling `StepCompleted.parameters` (`story:observatory-parameter-statistics`).
- `ess verify conform` suites: ESS 0.66.0 refuses this specification with `UnsupportedPrimitive:
  finite Binary64 is not admitted by the current conformance suite and codecs` (run on
  2026-10-21 by `ess verify conform synthesize --path .` in `spec/`), so the scenarios above are
  Rust tests against the ESS-generated JSON Schema.
