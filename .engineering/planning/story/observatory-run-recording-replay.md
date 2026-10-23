---
format: aep.planning-md/3
id: story:observatory-run-recording-replay
kind: story
status: implemented
title: Record a run file and replay it over the same event stream
summary: modelspec train writes a recorded run file and a replay serves its RunEvents identically to the live stream; settles the recorded-file half of the transport marker.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: story:observatory-live-event-stream
scope:
- confidence: inferred
  path: .gitignore
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: crates/modelspec-cli/src/lib.rs
- confidence: inferred
  path: crates/modelspec-cli/src/observe.rs
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/fixtures
- confidence: inferred
  path: crates/modelspec-cli/tests/observatory_record.rs
- confidence: cited
  path: crates/modelspec-observatory
- confidence: cited
  path: crates/modelspec-observatory/.ess-output/state.json
- confidence: cited
  path: crates/modelspec-observatory/schema
- confidence: inferred
  path: crates/modelspec-observatory/src/event.rs
- confidence: inferred
  path: crates/modelspec-observatory/src/lib.rs
- confidence: inferred
  path: crates/modelspec-observatory/src/record.rs
- confidence: cited
  path: crates/modelspec-observatory/src/server.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: website/docs/reference/cli.md
revision: 7
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T23:58:01Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "proposed", to: "active", at: "2026-10-22T23:58:01Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "active", to: "implemented", at: "2026-10-23T01:48:22Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"test_result":1,"review_outcome":2}}}
---
# Record a run file and replay it over the same event stream

## Context

`epic:training-observatory` (In scope, first bullet) makes the stream replayable from a recorded
run file "so the UI also works after the run". This story writes that file during
`modelspec train` and serves it back over the endpoint `story:observatory-live-event-stream`
settled, so a consumer cannot tell a replay from the live run except by timing.

## Open questions settled first

One `UNMAPPED:` marker of `spec/domains/observatory.yaml` stands in front of this story, named by
where it sits and a quoted fragment of its text, never by line number. It is settled before any
code, in that file, with `ess specify validate --path .` in `spec/` passing afterwards. The
settlement names its source (the operator, DESIGN.md or another ESS document); where none answers,
the story stops at that step and reports the question instead of choosing.

1. Domain-level marker `UNMAPPED (recorded file)`, text "the recorded run file a stream replays
   from (its format, the flag that writes it": the file format and extension (one JSON `RunEvent`
   per line is one candidate; a versioned header is another, as `.msckpt` has in DESIGN.md § 6.2),
   the flag that writes it, and the verb or flag that replays it (for example
   `modelspec observe <file>`). The live transport marker belongs to
   `story:observatory-live-event-stream`.

## Domain relations

- `TrainingRun` owns many `RunEvent`: `spec/domains/training.yaml`, entity
  `modelspec.training.TrainingRun`, relation `events` (owns, cardinality many, via
  `training_run_id`). A recording holds the events of exactly one run; a replay sends them in
  `sequence` order (`spec/domains/observatory.yaml`, entity `modelspec.observatory.RunEvent`, field
  `sequence`: "a replay sends events in this order").

## Scope

- `crates/modelspec-observatory/`: the recording writer and reader, replay through the existing
  SSE server.
- `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/src/lib.rs` and a new subcommand file
  if the settlement adds a verb; clap derive.
- `spec/domains/observatory.yaml` (the settlement; a file-format type if one is declared).
- `crates/modelspec-docs/`, `website/docs/reference/` through `task docs-generate`;
  `CHANGELOG.md` one line. `.gitignore` if the extension needs ignoring, as `*.msckpt` is.
- Tests under `crates/modelspec-cli/tests/` or `crates/modelspec-observatory/tests/`; a committed
  fixture recording of `examples/linear_regression.mspec` and the test that regenerates and
  compares it. The two-layer-relu fixture is `story:observatory-ui-live-loss-chart`.

## Acceptance

Conformance scenarios, each a named test run by `task ci` except scenario 1, which is the command
shown:

1. `spec/domains/observatory.yaml` no longer contains the marker `UNMAPPED (recorded file)`; a typed
   declaration that cites its source stands in its place, and `ess specify validate --path .` in
   `spec/` exits 0.
2. `observatory.record.matches-live`: one `modelspec train --backend cpu
   examples/linear_regression.mspec` invocation both serves and records; the events in the file
   equal, in order and field by field, the events a client read from the live stream.
3. `observatory.replay.matches-recording`: replaying the file serves the same events, same
   `sequence` and same `id:` lines, and each validates against the ESS-generated JSON Schema of
   `modelspec.observatory.RunEvent`.
4. `observatory.replay.refuses-damaged-file`: a truncated file, a file with a `sequence` gap and a
   file mixing two `training_run_id` values are each refused with a one-line error and exit 1
   (AGENTS.md exit codes), before anything is served.
5. `observatory.record.fixture-current`: the committed linear-regression fixture equals a fresh
   recording except for `training_run_id` and any field the settlement marks as per-invocation.

## Out of scope

- Comparing several recordings (epic, Out of scope).
- Recording parameter values or checkpoints (`spec/domains/checkpoint.yaml`, M1).
- The UI (`story:observatory-ui-live-loss-chart`).
