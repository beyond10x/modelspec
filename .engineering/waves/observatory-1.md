# Wave observatory-1: event stream and input adjoint

- Epic: `epic:training-observatory`; both units are the chain heads of the epic and run on the CPU
  reference only. No Vulkan, no device run.
- Approval: the operator asked to dispatch this wave on 2026-10-21 and approved the planning
  commit (option A) the same day; the wave, its unit commits, the merges and the merge into `main`
  are approved. On 2026-10-21 the operator also approved one push of `main` to `origin` after the
  wave merges into `main` (option A), carrying the planning commit and the wave together. No tag
  or release.
- Precondition: the planning work (DESIGN.md 0.7, `spec/domains/observatory.yaml`,
  `spec/domains/data.yaml`, the two epics, the eight observatory stories and their review-results)
  is committed on `main` first. That commit is not covered by this wave's approval.
- Rules: `aep:implementing` skill 0.23.2, wave mode. Selection read from
  `aep plan artifact waves` (aep 0.74.0) over typed scope entries written from each story's
  `## Scope` section.
- Integration branch: `wave/observatory-1`, from `main` at the planning commit. Managed tree id
  `modelspec-wave-observatory-1`.
- Units run in parallel (N = 2): their typed scopes share no path.
- Builds: every cargo command runs through `b10x-build-slot`; full test runs use
  `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only`. Each tree builds into its own
  `target/`. Pre-flight needs at least 20G free on `/`.
- Commits this approval covers: one per unit, the merge of each unit into `wave/observatory-1`, the
  closing store commit, the merge of `wave/observatory-1` into `main`, then one push of `main`. No
  tag or release.

## Selection (`aep plan artifact waves --kind story --status draft`, verbatim)

```
wave 1
  story:observatory-input-adjoint
  story:observatory-live-event-stream (inferred)
unassessed: story:observatory-activation-statistics
unassessed: story:observatory-architecture-graph
unassessed: story:observatory-parameter-statistics
unassessed: story:observatory-run-recording-replay
unassessed: story:observatory-saliency-gradient-x-input
unassessed: story:observatory-ui-live-loss-chart
1 wave(s), 0 collision(s), 6 unassessed
```

The six unassessed stories are not in this wave: each depends, through the epic's chain, on
`story:observatory-live-event-stream` or `story:observatory-input-adjoint`, and is scoped before
its own wave.

`(inferred)` on live-event-stream comes from three inferred entries: `Cargo.lock`,
`spec/domains/training.yaml`, `.github/workflows/check.yml`. None is in input-adjoint's scope.

## Coordinator decisions

- D1: live-event-stream does not edit `DESIGN.md`. § 7 is the three-crate starting sketch and
  already omits `modelspec-docs`; `DESIGN.md` and its revision line stay with input-adjoint alone.
- D2: live-event-stream installs `ess` in `.github/workflows/check.yml`, pinned to
  `spec/ess-inputs.yaml`, because its `task ci` gains `ess specify validate`.
- D3 (adversary pass 1, finding 1): the run event specification carries a non-finite float as
  JSON `null`; the CLI's stdout line names NaN or ±inf.
- D4 (adversary pass 1, finding 2): `train --serve` keeps serving after the run ends and, on
  SIGINT or SIGTERM, exits with the run's own status (0 finished, 1 failed).
- D5 (adversary pass 1, findings 3 and 4): finding 3 is no change, because scenario
  `observatory.stream.schema-current` is held by the `task schema-check` step of `task ci`, which
  the adversary saw fail on a planted schema edit, a deleted schema file and a spec summary edit;
  finding 4 is fixed by rewording the `AGENTS.md` line.

## Units

Worktrees live under the worktree manager's tree root; scratch under
`~/.cache/modelspec-wave-observatory-1/<unit>/scratch`; build directory `<tree>/target`.

| # | story | branch | tree id | head | stage |
|---|---|---|---|---|---|
| 1 | observatory-live-event-stream | impl/observatory-live-event-stream | modelspec-impl-observatory-live-event-stream | ab9ed27 | committed 99902f4; adversary pass 1: red, 4 findings (1 blocker), cases committed 18536b1; fix pass 1 paused, WIP committed 4b80ea4 |
| 2 | observatory-input-adjoint | impl/observatory-input-adjoint | modelspec-impl-observatory-input-adjoint | ab9ed27 | merged 0a760e8 (units 0d797e6, 85bc574); adversary pass 1: 0 defects, 1 test gap closed; VERIFIED |

Integration tree: `modelspec-wave-observatory-1` on `wave/observatory-1` at `ab9ed27`. Both
stories moved draft → proposed → active there. Pre-flight: 29G free on `/`; one measured build
directory 1.9G; N = 2. Unit briefs: `~/.cache/modelspec-wave-observatory-1/<unit>/scratch/brief.md`.

Subagent types: `aep:story-scoper` (done), `aep:implementor`, `aep:adversary`.

## Closed 2026-10-22

- Unit 1: fix pass 1 `627c650`, adversary pass 2 cases `7668994` (3 findings: 1 warning, 2
  notes), fix pass 2 `91624a0`; merged into `wave/observatory-1` as `f4b3968`. Pass-1 outcomes
  fixed, fixed, no-op, fixed; pass-2 outcomes fixed ×3.
- D6 (coordinator): the `modelspec-docs` CLI page test, red since `99902f4` because `--serve`
  added `[OPTIONS]` and a Required column, is fixed in `627c650`.
- D7 (adversary pass 2, F1): the stdout promise is narrowed to "every printed step whose loss is
  null names NaN or Inf"; F2 and F3: the signal handler is installed before the last stdout line
  and covers SIGHUP.
- `task ci` on `f4b3968`: exit 0. Both stories moved active → implemented.
- Merged with `main` at `c1c47c7` (wave cpu-perf-1) and `plan/observatory`, and DESIGN.md
  revision 0.9 (ADR-007), before the PR into `main`.

## Paused 2026-10-21

The operator paused modelspec on 2026-10-21. Nothing is merged into `main`; `main` is not pushed.
Pushed: `wave/observatory-1` (this page), `impl/observatory-input-adjoint` at `85bc574`,
`impl/observatory-live-event-stream` at `4b80ea4`. The unit briefs are in
`.engineering/paused/observatory-live-event-stream/`.

- Done: unit 2 merged and verified. Unit 1 fix pass 1: F4 (`AGENTS.md` line) done. F1 is half
  done: `spec/domains/observatory.yaml` carries non-finite floats as `null` and validates (7
  files), and new tests are in `modelspec-observatory/tests/stream.rs`. F2 is half done: the spec
  text is updated and the SIGTERM and SIGINT test is in `modelspec-cli/tests/observatory_stream.rs`.
  None of the new tests has run.
- Open on unit 1: the serializer in `event.rs`, `task schema-generate`, the `ctrlc` dependency
  and the signal handling in `train.rs`, then the gate, adversary pass 2, and recording
  `review_outcome` per finding. `task ci` is expected red at `4b80ea4` because the committed
  schema predates the spec.
- Known conflict: when the schema admits `null`, the adversary case
  `the_schema_refuses_what_the_spec_forbids` (its "loss null" check) goes red. It reflects the
  spec before D3, so it is changed in the fix pass, and the change is recorded beside finding 1.
- Then the close: `CHANGELOG.md` line, full `task ci`, scope write-back, both stories moved to
  implemented, the closing store commit, the merge into `main`, and one push of `main`.
- First step on resume: start a new managed tree from `impl/observatory-live-event-stream`, run
  the two adversary cases with `src/` and `schema/` as committed, to get their red result. Then
  run the new tests red, and then make the Rust changes.
