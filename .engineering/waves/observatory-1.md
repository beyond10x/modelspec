# Wave observatory-1: event stream and input adjoint

- Epic: `epic:training-observatory`; both units are the chain heads of the epic and run on the CPU
  reference only. No Vulkan, no device run.
- Approval: the operator asked to dispatch this wave on 2026-10-21 and approved the planning
  commit (option A) the same day; the wave, its unit commits, the merges and the merge into `main`
  are approved. No push, tag or release.
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
  closing store commit, the merge of `wave/observatory-1` into `main`. No push, tag or release.

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

## Units

Worktrees live under the worktree manager's tree root; scratch under
`~/.cache/modelspec-wave-observatory-1/<unit>/scratch`; build directory `<tree>/target`.

| # | story | branch | tree id | head | stage |
|---|---|---|---|---|---|
| 1 | observatory-live-event-stream | impl/observatory-live-event-stream | modelspec-impl-observatory-live-event-stream | — | proposed |
| 2 | observatory-input-adjoint | impl/observatory-input-adjoint | modelspec-impl-observatory-input-adjoint | — | proposed |

Subagent types: `aep:story-scoper` (done), `aep:implementor`, `aep:adversary`.
