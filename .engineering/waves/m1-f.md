# Wave m1-f

Serves `epic:pytorch-10x-mnist` (GPU lane): the two matmuls that the wave m1-e profile shows as the
longest dispatches of the step.

## Selection

- `story:m1-f-forward-matmul-tiled`, then `story:m1-f-weight-grad-matmul-tiled`. They share
  `emit.rs`, `schedule.rs`, `kernel.rs`, DESIGN.md and CHANGELOG.md, so they run in sequence; the
  second depends on the first.

## Approval

- `approval-record:m1-f-wave`: approved by the chef orchestrator on 2026-10-22, with the thresholds,
  shared conditions and miss rule of the proposal.
- `approval-record:m1-waves-and-device-runs`: device runs on the RTX 3090.
- `approval-record:m1-tolerance-numerics-class`: per-operator tolerances in every GPU test.
- Commits: one per unit, the merges into `wave/m1-f`, the closing store commit and the fast-forward
  of `main`, all through the bot. No tag, no release; superseded on 2026-10-23 by Timo's request
  for release 0.5.0 (see Resumed).

## Starting point

`main` `e77066f` (release 0.4.0, the m1-e code of `7a488c1`). ADR-007 method, m1-e close: 40.225 µs
per step, median of 5 runs of 144,000 steps (39.647 to 40.619), mean P0 1935 to 1951 MHz; 14
dispatches per step. 1,800-step stdout sha256 `b9bb028a…6904`.

Wave m1-f profile on `7a488c1`, 144,000 steps, 3 runs after a warm-up, P0 in 165 samples at a mean
of 1951 MHz, mean µs per site over the 3 runs:

| position | kernel | mean µs |
|---|---|---|
| 0 | `k0` `0f20c50b`, matmul + ReLU, `[100,784] x [784,64]` | 18.372 / 19.279 / 18.545 |
| 17 | `k11` `433c82fb`, matmul, `[784,100] x [100,64]` | 6.697 / 7.218 / 6.957 |
| 4 | loss sum `d588cbc2` | 4.251 / 4.459 / 4.133 |
| 13 | `k8` `a94b7d6e`, matmul, `[64,100] x [100,10]` | 3.466 / 3.555 / 3.498 |
| 15 | `cbd56b2b` | 3.262 / 3.327 / 3.221 |
| 2 | `k1` `cba0fd87`, matmul, `[100,64] x [64,10]` | 2.939 / 2.959 / 2.824 |
| 10 | `k6` `1902a24e`, matmul, `[100,10] x [10,64]` | 2.706 / 2.821 / 2.695 |
| other 7 sites | | 1.848 to 2.583 each |

## Pre-flight

- `df -h /`: 50G free after the m1-e and release trees were finished with `--discard-cache`.
- Builds go through `b10x-build-slot`; a timed GPU run gets the GPU alone.

## Units

| # | story | branch | tree id | build dir | stage |
|---|---|---|---|---|---|
| 1 | m1-f-forward-matmul-tiled | impl/m1-f-forward-matmul-tiled | modelspec-impl-m1f-fwd2 | `<tree>/target` | merged into `wave/m1-f` at `5efefae`; head `07f6edf` |
| 2 | m1-f-weight-grad-matmul-tiled | impl/m1-f-weight-grad-matmul-tiled | none | | moved to the next wave |

Integration: `wave/m1-f`, tree `modelspec-wave-m1f2`, from `main` `e77066f`.
Scratch: `~/.cache/modelspec-wave-m1-f/<story>/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.

## Held, 2026-10-22

- Builds paused for disk at the orchestrator's request; both trees finished, both branches pushed.
- Unit 1: `a7d7903` on `impl/m1-f-forward-matmul-tiled` gives `k0` median 8.732 µs and 28.378 µs per
  step at a mean P0 of 1953 MHz, sha256 `b9bb028a…6904` in 5 of 5 runs.
- `review-result:m1-f-forward-adversary-1`: two blockers in `emit.rs:1538`. The unrolled tile loop
  passes the SPIR-V id bound near `k = 290,000`, and pipeline creation takes 188.7 s at
  `k = 60,000`. The adversary's two test files are the second commit on the unit branch and are red
  until the fix.
- Resume: recreate both trees from the pushed branches, then dispatch fix round 1 with
  `~/.cache/modelspec-wave-m1-f/brief-forward-fix1.md`. Unit 2 follows the merged unit 1.

## Resumed, 2026-10-23

- Timo asked for a commit and a new version before the next wave. The wave closes with unit 1
  alone; unit 2 joins the next wave. Release 0.5.0 follows the merge of `wave/m1-f` into `main`,
  through the `AGENTS.md` § Releases procedure, on Timo's request.
- Trees recreated from the pushed branches: `modelspec-impl-m1f-fwd2` at `41868b8`,
  `modelspec-wave-m1f2` at `9b4518b`. Build floor 25G free, at the orchestrator's request.
- Fix round 1, `ab375a5`: the k-tiles run in a loop of 4-tile passes, so a module's size does not
  grow with `k`; a 2-row tile from 256 one-row workgroups. `k0` 8.302 / 8.279 / 8.352 µs, 27.282 µs
  per step (27.267 to 27.435), median of 5 runs of 144,000 steps at a mean P0 of 1957 to 1965 MHz;
  sha256 `b9bb028a…6904` in 5 of 5 runs.
- `review-result:m1-f-forward-adversary-2`: 1 warning, test only. The size-bound slack of 4
  constants in `spirv_val.rs` is 5 ids short at `[208, k] x [k, 2]`. Correction 2, `07f6edf`,
  derives the slack from the emitter's k-dependent constants; no change under `src/`.
- Findings per pass: 5, then 1; 0 carried.

## Close, 2026-10-23

- `wave/m1-f` merge `5efefae` (no-ff of `07f6edf`): `task ci` exit 0, 1268 passed, 0 failed, 90
  ignored; `aep plan artifact validate`: 135 artifacts, valid.
- `story:m1-f-forward-matmul-tiled` implemented (test_result 1, review_outcome 2).
- `story:m1-f-weight-grad-matmul-tiled` stays active and moves to the next wave.
- `main` moves to the `wave/m1-f` head after its pull request's required checks pass; release 0.5.0
  follows.
