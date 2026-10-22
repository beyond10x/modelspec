# Wave m1-d

Serves `epic:m1-vulkan-execution` and, through it, `epic:pytorch-10x-mnist` (GPU lane).

## Selection

- Unit: `story:m1-vulkan-train`. N is 1.
- Its `depends_on` stories (`m1-mlp-example`, `m1-vulkan-reduce-matmul`, `m1-checkpoint-cpu`,
  `observatory-live-event-stream`) are implemented on `main` `2a5481e` (wave m1-c).
- `DESIGN.md` revision 0.16 for this unit.
- After the merge: the 784-64-10 µs/step measurement on the RTX 3090, with the GPU alone.
- Next: `m1-compile-artifact`, `m1-vulkan-checkpoint-infer` and `m1-vulkan-profile`, all of which
  depend on this story.

## Approval

- `approval-record:m1-waves-and-device-runs`: the M1 waves and device runs on the RTX 3090.
- `approval-record:m1-b-start`: start each next M1 wave without a further go.
- `approval-record:m1-tolerance-numerics-class`: every GPU test names its tolerance per operator.

## Pre-flight

- `df -h /`: 72G free.
- Builds go through `b10x-build-slot`; a timed GPU run gets the GPU alone.

## Units

| # | story | branch | tree id | stage |
|---|---|---|---|---|
| 1 | m1-vulkan-train | impl/m1-vulkan-train | modelspec-impl-train | merged 2c34824 |

Integration: `wave/m1-d`, tree `modelspec-wave-m1d`, from `main` `2a5481e`.
Scratch: `~/.cache/modelspec-wave-m1-d/<story>/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.

## Close

| check on `2c34824` | result |
|---|---|
| adversary pass 1 | NEEDS-CHANGE, 3 warnings and 2 notes, all fixed in `2f18933` |
| `task ci` | 1219 passed, 0 failed, exit 0 |
| RTX 3090 device tests, validation on | exit 0, class tolerance, 1 host wait per run |
| `train examples/mnist_mlp.mspec --backend vulkan --stats`, release, validation off | 1 host wait, 8 submits, 18 dispatches per step, 520864 bytes up, 210468 down |

## 784-64-10 step time

15 back-to-back release runs, validation off, no other compute process: 139.2 to 158.2 µs per
step, median 145.3. The GPU stayed at P3, 780 MHz of 2130 MHz, during every run. Target of
`epic:pytorch-10x-mnist`: 31.4 µs per step.

A second adversary pass was not run: the fix adds the adversary's own cases, and each of the
three other changes has a planted-defect run.
