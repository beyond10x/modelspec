---
format: aep.planning-md/3
id: story:observatory-ui-live-loss-chart
kind: story
status: active
title: 'Observatory UI: live loss and metric charts in Vue, types generated from Rust'
summary: A Vue 3 + TypeScript + Vite + pnpm UI served by the Rust server draws loss and metric curves live and from a replay; its server types are generated from Rust and drift-checked.
relations:
- decomposes: epic:training-observatory
- serves: vision:modelspec
- depends_on: story:observatory-live-event-stream
- depends_on: story:observatory-run-recording-replay
scope:
- confidence: cited
  path: .github/workflows/check.yml
- confidence: inferred
  path: .gitignore
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: inferred
  path: Cargo.lock
- confidence: inferred
  path: Cargo.toml
- confidence: cited
  path: README.md
- confidence: cited
  path: Taskfile.yml
- confidence: inferred
  path: crates/modelspec-cli/Cargo.toml
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures
- confidence: cited
  path: crates/modelspec-observatory/
- confidence: inferred
  path: crates/modelspec-observatory/.ess-output/state.json
- confidence: inferred
  path: crates/modelspec-observatory/schema/
- confidence: inferred
  path: crates/modelspec-observatory/src/event.rs
- confidence: cited
  path: crates/modelspec-observatory/src/server.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: ui/
- confidence: cited
  path: website/docs/index.md
- confidence: inferred
  path: website/docs/reference/cli.md
- confidence: inferred
  path: website/docs/reference/crates.md
revision: 27
transitions:
- {from: "draft", to: "proposed", at: "2026-10-23T08:16:30Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"review_outcome":3}}}
- {from: "proposed", to: "active", at: "2026-10-23T08:16:30Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"review_outcome":3}}}
---
# Observatory UI: live loss and metric charts in Vue, types generated from Rust

## Context

The browser half of `epic:training-observatory`: a Vue 3 + TypeScript app, built with Vite and
pnpm (workspace rule, 2026-10-21), served by the Rust server of
`story:observatory-live-event-stream`, that subscribes to the stream with `EventSource` and draws
the loss and every metric per step. It works the same against a live run and a replay
(`story:observatory-run-recording-replay`). Owns the second epic Acceptance bullet ("The UI renders
the linear-regression and two-layer-relu runs live and from a recording") and the type-sharing
rule: every type the UI reads from the server is generated from the Rust types in
`crates/modelspec-observatory/` and drift-checked in `task ci` (epic, In scope, third bullet).

The two-layer-relu model is `crates/modelspec-docs/models/two_layer_relu.mspec` (`TwoLayer`,
200 steps, seed 7); there is no copy under `examples/`.

## Open questions settled first

None of the `UNMAPPED:` markers. Two build decisions are made in this story and recorded in
`AGENTS.md`: where the app lives (`ui/` at the repository root is the default; the precedent is
`uilab/widget/`), and how the server finds the built app. Constraint on the second:
`task build` and `task test` keep working on a machine without Node or pnpm, as they do today
(no `build.rs` or cargo step may call pnpm); `AGENTS.md` currently says nothing but `website/`
uses Node, and that line changes here.

## Domain relations

- `TrainingRun` owns many `RunEvent`: `spec/domains/training.yaml`, entity
  `modelspec.training.TrainingRun`, relation `events` (owns, cardinality many, via
  `training_run_id`). The UI shows one run: the events of one `training_run_id` in `sequence` order.

## Scope

- `ui/` (new): Vue 3 + TypeScript + Vite + pnpm app; `ui/src/generated/` holds the generated types
  and is never edited by hand.
- `crates/modelspec-observatory/`: a Rust generator that writes the TypeScript types of every type
  the server sends (`RunEvent` and what it contains) with a `--check` mode; serving the built app at
  `/` beside the stream.
- `Taskfile.yml`: a `ui-types-check` task in `task ci`; a `ui` task (`pnpm install --frozen-lockfile`,
  `pnpm build`, the UI tests) and the headless browser test, wired into `task ci`.
- A committed fixture recording of `crates/modelspec-docs/models/two_layer_relu.mspec`, written by
  the recording flag of `story:observatory-run-recording-replay`, beside its linear-regression
  fixture.
- A committed fixture recording whose `StepCompleted` events carry a `MetricSample` named `accuracy`,
  written by this story (a test helper that emits `RunEvent`s; no M0 producer fills `metrics`), for
  scenario 6.
- `crates/modelspec-cli/src/train.rs` only if the serve flag needs a UI option.
- A Rust integration test that drives a headless browser (crate chosen here; the harness is Rust,
  workspace rule). I do not know whether the CI runner of `check.yml` has a Chromium; the story
  finds out and adds it to the workflow if not.
- `AGENTS.md` (Commands, State), `README.md` (how to open the UI), `CHANGELOG.md` one line;
  `website/docs/` hand-written guide page if one describes training.

## Acceptance

Conformance scenarios 1 to 6 are named tests run by `task ci`; scenario 7 is a named check that
`task ci` does not run, because `task ci` itself needs pnpm.

1. `observatory.ui.types-in-sync`: the TypeScript under `ui/src/generated/` equals what the Rust
   generator writes now; a changed field in `crates/modelspec-observatory/` without regeneration
   fails `task ci`.
2. `observatory.ui.live-linear-regression`: with `modelspec train --backend cpu
   examples/linear_regression.mspec` serving, a headless browser opens the UI and the last point of
   the loss chart equals the loss of the last `StepCompleted` the stream sent, and the chart has one
   point per `StepCompleted`.
3. `observatory.ui.live-two-layer-relu`: the same for
   `crates/modelspec-docs/models/two_layer_relu.mspec`.
4. `observatory.ui.replay-linear-regression`: the same check against a replay of the
   linear-regression fixture recording, with no training process running.
5. `observatory.ui.replay-two-layer-relu`: the same check against a replay of the two-layer-relu
   fixture recording, with no training process running.
6. `observatory.ui.metric-charts`: a recording whose `StepCompleted` events carry a
   `MetricSample` named `accuracy` draws an `accuracy` chart whose last point equals the last
   sample; a recording with no metrics draws the loss chart only.
7. `observatory.ui.builds-without-node`: from the repository root, `sh ui/scripts/build-without-node.sh`
   exits 0. The script, committed by this story, builds a fresh directory of symlinks and makes it
   the whole PATH, so it runs on any machine, whatever `/usr/bin` holds:
   `bin=$(mktemp -d)`; for each of `cargo rustc rustdoc cc ld as ar task sh env cat mkdir rm`,
   `ln -s "$(command -v "$t")" "$bin/$t"`; then
   `PATH="$bin" "$bin/sh" -c '! command -v node && ! command -v pnpm && task build test'`. The
   implementer may add a tool `task build test` needs to that list, never `node`, `npm`, `npx` or
   `pnpm`. It guards that no cargo build step calls pnpm. `task ci` does not run it; the implementer
   and the reviewer run it and record its output as evidence.

## Out of scope

- The architecture, statistics and saliency views (their own stories).
- A metric producer: M0 computes no metric besides the loss, so scenario 6 uses the `accuracy` fixture this story writes.
- Authentication, hosting, remote access, run comparison (epic, Out of scope).
