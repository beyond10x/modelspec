# Wave m1-e

Serves `epic:pytorch-10x-mnist` (GPU lane) and `epic:m1-vulkan-execution` (DESIGN.md § 9
checkbox 9).

## Selection

- Stage 1: `story:m1-vulkan-profile`, alone. Its per-dispatch times rank the fusion stories.
- Stage 2, after the profile: up to three fusion stories in parallel, one implementor each,
  drafted and sized from the profile's numbers: matmul with bias and ReLU as one kernel and the
  ReLU gradient folded into the backward matmul; every parameter update of a step in one
  dispatch; the `mse` loss sum as a workgroup tree reduction instead of one invocation. A
  candidate the profile shows saves under 5 % of the step time is dropped, and the drop reported.
- `story:m1-vulkan-profile` no longer depends on `story:m1-compile-artifact`: the guide's
  built-kernel blocks moved to that story's acceptance 7.
- DESIGN.md revision 0.17 in the open commit: ADR-007 names the working target, 31.4 µs per step,
  and the GPU timing method. The profile story takes 0.18.

## Approval

- Wave m1-e approved by the chef orchestrator on 2026-10-22, as proposed: the profile first, then
  the fusion stories in parallel; m1-f waits for the profile and a further approval.
- `approval-record:m1-waves-and-device-runs`: the M1 waves and device runs on the RTX 3090.
- `approval-record:m1-tolerance-numerics-class`: every GPU test names its tolerance per operator.
- Sizing from the profile, accepted by the chef orchestrator on 2026-10-22: the loss sum kept as
  `story:m1-loss-sum-staged` (about 27 µs); the relu epilogue (about 2 %) and the fused SGD update
  (about 4.6 %) each under 5 %, merged with the constant fold into `story:m1-fuse-elementwise`
  (about 6.6 %); the relu-grad fold dropped (its group is set by longer kernels). The two run in
  sequence, loss sum first, as they share `emit.rs`, `schedule.rs` and `kernel.rs`.
- m1-f (the two large matmuls, 23.9 µs profiled) approved, to be proposed after m1-e merges and its
  median is re-timed by the ADR-007 method; each m1-f story carries a measured acceptance threshold.
- `story:m1-loss-sum-staged` acceptance 3 restated to "at most 6 µs and not the longest dispatch"
  (measured 5.620 µs), accepted by the chef orchestrator on 2026-10-22.
- Closing re-time: base (`ec431b1`) and the `wave/m1-e` head are timed in one session at the same
  sampled clock, each with its mean P0 MHz in the record; that pair is the m1-e result, not a unit's
  own before/after figure.

## Starting point

`main` `ec431b1` on the RTX 3090, ADR-007 timing method: 86.1 µs per step median, 85.0 to 86.4
over 5 runs of 72,000 steps after a 6.5 s warm-up; P0 in 102 of 162 clock samples at a mean of
1744 MHz. 18 dispatches per step. The working target is 31.4 µs per step.

## Pre-flight

- `df -h /`: 76G free.
- Builds go through `b10x-build-slot`; a timed GPU run gets the GPU alone.

## Units

| # | story | branch | tree id | stage |
|---|---|---|---|---|
| 1 | m1-vulkan-profile | impl/m1-vulkan-profile | modelspec-impl-profile | merged 1ac97bf |
| 2 | m1-loss-sum-staged | impl/m1-loss-sum-staged | modelspec-impl-losssum | merged 6056e81 |
| 3 | m1-fuse-elementwise | impl/m1-fuse-elementwise | modelspec-impl-fuse | merged 170c027 |

Integration: `wave/m1-e`, tree `modelspec-wave-m1e`, from `main` `ec431b1`.
Scratch: `~/.cache/modelspec-wave-m1-e/<story>/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.

## Close

Closing re-time on 2026-10-22, release builds of `ec431b1` and `7a488c1` alternated in one session
on the RTX 3090, validation off, no other compute process, after a warm-up run of each: P0 in 525 of
526 clock samples.

| run | `ec431b1` µs per step | `7a488c1` µs per step | mean P0 MHz (base / wave) |
|---|---|---|---|
| 144,000 steps, 5 runs each, median | 69.184 (68.659 to 69.768) | 40.225 (39.647 to 40.619) | 1935 to 1965 / 1935 to 1951 |
| 1,800 steps, 5 runs each, median | 69.762 (69.012 to 71.616) | 42.745 (40.182 to 43.294) | 1935 / 1935 |
| dispatches per step (`--stats`) | 18 | 14 | |
| stdout of the 1,800-step run, sha256 | `b9bb028a…6904` | `b9bb028a…6904` | |

The m1-e result is 69.184 to 40.225 µs per step (−41.9 %) at the same clock. On the 1,800-step run,
42.745 µs per step with batch 100 is 2,339,455 images/s against the ADR-007 gate of 1,927,410; the
gate's accuracy and final-weight conditions were not measured here. The working target is 31.4 µs
per step.
