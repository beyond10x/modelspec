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
| 1 | cpu-matmul-avx2-blocking | impl/cpu-matmul-avx2-blocking | modelspec-impl-avx2 | `target/` in the tree | `~/.cache/modelspec-wave-cpu-perf-2/cpu-matmul-avx2-blocking/scratch/` | f250dd5 | merged (984c022) |

Integration: `wave/cpu-perf-2`, tree `modelspec-wave-cpu2`, from `main` `dc7c48e`.

Dispatch: `aep:implementor`, then `aep:adversary`.

## Closed 2026-10-22

- Unit commit `f250dd5`, merged into `wave/cpu-perf-2` as `984c022`.
- Adversary: pass 1 found 2 (one blocker, release-mode count overflow reaching the unsafe kernel;
  one note, an unused `mut` on non-x86 targets), both fixed; pass 2 found 0.
  `aep plan artifact findings`: carried 0, new 0, resolved 2.
- Claim VERIFIED: `mnist_shape --bench`, 9 interleaved pairs on core 3, change median 405 µs/step
  (390–668) against base 1,207 (1,186–1,933), 2.98×; target 600. Weights blake3 `805f64bd…02df`
  and loss bits `0x3dbbe63b` on all 18 runs. PyTorch on one CPU thread: 526 µs/step.
- Gate on `984c022`: `b10x-build-slot task ci` exit 0, and the release run of the three matmul test
  files exit 0; 47 test binaries, 657 passed, 0 failed. Story moved to `implemented`.
- Planted defects: an FMA failed 6 of 13 tests; a dropped last partial column failed 7 of 13.
- Not done: the right-operand transpose is still copied; no blocking over `k`.

Agent cost:

| Agent | Tokens | Tool uses | Time |
|---|---|---|---|
| Scoper | 61,399 | 21 | 122 s |
| Implementor | 161,993 + 41,280 | 91 + 20 | 1,936 s + 473 s |
| Adversary | 79,998 + 97,276 | 27 + 10 | 279 s + 120 s |
