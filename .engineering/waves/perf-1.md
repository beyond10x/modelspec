# Wave perf-1

Serves `vision:modelspec` and the operator's goal of 2026-10-24: the performance findings of the
two code reviews (of 0.5.0 and of 0.7.1) implemented, then a new version cut. Skill
`aep:implementing` 0.23.2, wave mode.

## Selection

- `aep plan artifact waves --status draft --format json` at the review-drafts head: 22 waves,
  785 collisions, 17 unassessed, 0 cycles. Among the 15 performance stories, the collision
  pairs are recorded in `approval-record:perf-wave-1`; the five below collide with none of each
  other and have no open `depends_on`.
- Every unit's typed scope is cited; `vulkan-step-counters` carries inferred entries for the
  guide and the CLI reference.
- Dispatch: `aep:implementor` (Opus) per unit, then `aep:adversary` (Opus) per green unit.

## Approval

`approval-record:perf-wave-1`: the operator's words of 2026-10-24, "I accept all you want, you act
on behalf of me", under the goal "performance issues addressed from review and new version cut".
Commits the approval covers: 5 unit commits, the adversary cases on the unit branches, the merges
into `wave/perf-1`, the closing store commit, the fast-forward of `main` to the wave head after
its checks. The release is the step after the performance waves, through the bot.

## Pre-flight

- `df -h /`: 151G free at the start (floor 20G, `b10x-build-slot` pauses builds under 25G).
- One measured build: the primary checkout's `target/` is 4.3G (debug and test profiles).
- Builds go through `b10x-build-slot`, 3 slots.
- No other managed tree of this repository besides `modelspec-review-drafts` (lease
  `coord-9127ff35`, to be finished after PR 29 lands) and `wt-16f21ff40991` (not this session's).

## Units

| # | story | branch | tree id | build dir | scratch | head | stage |
|---|---|---|---|---|---|---|---|
| 1 | cpu-prepared-training-step | impl/cpu-prepared-training-step | modelspec-impl-prepared | `target/` in the tree | `~/.cache/modelspec-wave-perf-1/prepared/scratch/` | | dispatched |
| 2 | dense-step-reuses-batch-buffers | impl/dense-step-reuses-batch-buffers | modelspec-impl-dense | `target/` in the tree | `~/.cache/modelspec-wave-perf-1/dense/scratch/` | | dispatched |
| 3 | vulkan-step-counters | impl/vulkan-step-counters | modelspec-impl-counters | `target/` in the tree | `~/.cache/modelspec-wave-perf-1/counters/scratch/` | | dispatched |
| 4 | cpu-matmul-output-skips-zero-fill | impl/cpu-matmul-output-skips-zero-fill | modelspec-impl-zerofill | `target/` in the tree | `~/.cache/modelspec-wave-perf-1/zerofill/scratch/` | | dispatched |
| 5 | parser-span-lookup-not-linear-in-column | impl/parser-span-lookup-not-linear-in-column | modelspec-impl-span | `target/` in the tree | `~/.cache/modelspec-wave-perf-1/span/scratch/` | | dispatched |

Integration: `wave/perf-1`, tree `modelspec-wave-perf-1`, from `main` `5632eafd9c4f24e565a0e24d7682d949efdc511e`. Briefs:
`~/.cache/modelspec-wave-perf-1/<unit>/brief.md`.
