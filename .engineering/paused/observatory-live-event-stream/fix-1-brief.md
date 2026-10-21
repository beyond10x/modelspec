# Fix brief 1: story:observatory-live-event-stream, after adversary pass 1

Read the unit brief first, in full, in this directory: `brief.md`. Its triple, file assignment,
invariants and gate bind this pass unchanged, with the two additions under "Files" below.

Branch head: `18536b1` "observatory: adversary cases for the live event stream", on `99902f4`.
Two committed cases are red on it:

- `crates/modelspec-observatory/tests/adversary_schema.rs::a_non_finite_loss_frame_is_schema_valid`
- `crates/modelspec-cli/tests/observatory_stream_adversary.rs::a_diverging_run_streams_only_schema_valid_events`

The adversary's full report is `adversary-pass-1-report.md` in this directory, its logs are in
`adv/`; the findings to fix are below. Do not edit or weaken the adversary's cases. If one of them asserts something these
decisions contradict, say so in the report and do not change it.

## Finding 1 (blocker): a non-finite loss breaks the schema

`crates/modelspec-observatory/src/server.rs:603`: serde_json frames a NaN or ±inf f32 as `null`;
the committed RunEvent schema requires a number, so 45 of 53 events of an `lr = 1000.0` run fail.

Coordinator decision (2026-10-21): the specification carries a non-finite value as JSON `null`.

- In `spec/domains/observatory.yaml`, make `StepCompleted.loss` admit `null`, with a summary saying
  `null` means the loss was NaN or ±inf and that the CLI's `step <k> loss <v>` stdout line names
  which. Do the same for every other float an event carries (parameter values or statistics, if
  `StepCompleted.parameters` holds floats). Use the ESS construct that projects to a JSON Schema
  admitting `null` (check `ess specify formats` and the pinned release in `spec/ess-inputs.yaml`);
  run `ess specify validate --path .` from `spec/`, then `task schema-generate` and
  `task schema-check`.
- The Rust side must emit exactly what the schema now says, for all such fields.
- Both red cases above must turn green without being edited.

## Finding 2 (warning): `--serve` never exits 0 or 1

The settlement "keeps serving until interrupted" cites no source, and a finished run and an
E-DATA-003 run both exit 143 on SIGTERM.

Coordinator decision (2026-10-21), and the source to cite in the spec: "wave observatory-1
coordinator, 2026-10-21". After the run ends the process keeps serving, so a late client still
reads the whole stream. On SIGINT or SIGTERM it stops serving and exits with the run's own status:
0 for a run that finished, 1 for a run that failed (the same code plain `train` gives). Stdout and
stderr stay as they are.

- Write the source into the spec's settlement text in place of the story-question citation.
- Signal handling: use the smallest maintained crate that covers SIGINT and SIGTERM (for example
  `ctrlc` with its `termination` feature) in `modelspec-observatory` or `modelspec-cli`, never in
  `modelspec-core`; `task core-no-vulkan` must stay green.
- Add a CLI test: a finished run under `--serve` sent SIGTERM exits 0, a failing run (E-DATA-003)
  exits 1. Show it red first (exit 143) and paste the red summary in the report.

## Finding 4 (note): the AGENTS.md line overclaims

The `task schema-check` line in `AGENTS.md` says it fails when `schema/` differs; a hand-added
extra file passes. Reword that one line to what the check does (it fails when a generated file
differs from the specification or is missing). You may edit that line of `AGENTS.md` in this pass.

## Finding 3

Not yours: recorded by the coordinator as no change.

## Files

In addition to the unit brief's **yours**: the one `task schema-check` line of `AGENTS.md`, and the
generated `schema/` tree only through `task schema-generate`.

## Report

The unit brief's six-line header, then: the red runs (the two adversary cases at `18536b1`, and
your SIGTERM test before the fix), the gate with each command's own summary line and exit status,
the files changed, and per finding what changed.
