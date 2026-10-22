# Wave cpu-perf-2

Serves `epic:pytorch-10x-mnist`, CPU lane; the story serves `vision:modelspec`.

## Selection

- Candidate: `story:cpu-matmul-avx2-blocking` (draft; `depends_on story:cpu-matmul-row-order`,
  implemented). N is 1: no other draft story is on the CPU lane, and the M1 stories need device
  runs, which are the operator's.
- `aep plan artifact waves --kind story --status draft` (aep 0.74.0) before scoping: the story was
  unassessed; no cycle. After scoping its typed scope sits in `crates/modelspec-runtime`.
- Scope confidence high, from `story-scoper`; the section in the story marks each line cited or
  inferred.

## Approval

The resume message of b10x-chef-orchestrator on 2026-10-22, under exception 6 of the operator's
grants, approved continuing "then the next wave from your store". The grant covers: the unit
commit(s) on `impl/cpu-matmul-avx2-blocking` through the bot, the adversary's cases on that
branch, the merge into `wave/cpu-perf-2`, the closing store commit, and landing `wave/cpu-perf-2`
on `main` through a PR. No tag or release.

## Pre-flight

- `df -h /`: 100G free at the start.
- One measured build: the release tree's `target/` was 1,731 MiB at finish.
- Builds go through `b10x-build-slot`, one cargo run at a time.
- Bench rule: base `dc7c48e` and change interleaved on one pinned core, at least 5 pairs.

## Units

| # | story | branch | tree id | build dir | scratch | head | stage |
|---|---|---|---|---|---|---|---|
| 1 | cpu-matmul-avx2-blocking | impl/cpu-matmul-avx2-blocking | modelspec-impl-avx2 | `target/` in the tree | `~/.cache/modelspec-wave-cpu-perf-2/cpu-matmul-avx2-blocking/scratch/` | dc7c48e | planned |

Integration: `wave/cpu-perf-2`, tree `modelspec-wave-cpu2`, from `main` `dc7c48e`.

Dispatch: `aep:implementor`, then `aep:adversary`.
