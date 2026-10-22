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
  of `main`, all through the bot. No tag, no release.

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
| 1 | m1-f-forward-matmul-tiled | impl/m1-f-forward-matmul-tiled | modelspec-impl-m1f-forward | `<tree>/target` | planned |
| 2 | m1-f-weight-grad-matmul-tiled | impl/m1-f-weight-grad-matmul-tiled | modelspec-impl-m1f-wgrad | `<tree>/target` | planned |

Integration: `wave/m1-f`, tree `modelspec-wave-m1f`, from `main` `e77066f`.
Scratch: `~/.cache/modelspec-wave-m1-f/<story>/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.
