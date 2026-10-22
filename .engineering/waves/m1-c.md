# Wave m1-c

Serves `epic:m1-vulkan-execution` and, through it, `epic:pytorch-10x-mnist` (GPU lane).

## Selection

- Units: `story:m1-vulkan-reduce-matmul`, `story:m1-checkpoint-cpu`. N is 2, in parallel.
- Their `depends_on` stories (`m1-vulkan-elementwise`, `m1-kernel-plan`,
  `observatory-live-event-stream`) are implemented on `main` `23adba4` (wave m1-b).
- Shared files: `DESIGN.md` and `CHANGELOG.md`, both append-only surfaces; the coordinator
  resolves them at merge. Revision 0.14 for reduce-matmul, 0.15 for checkpoint.
- Next wave: m1-d (`m1-vulkan-train`), then the 784-64-10 µs/step measurement.

## Approval

- `approval-record:m1-waves-and-device-runs`: the M1 waves and device runs on the RTX 3090.
- `approval-record:m1-b-start`: start each next M1 wave without a further go.
- `approval-record:m1-tolerance-numerics-class`: every GPU test names its tolerance per operator.

## Pre-flight

- `df -h /`: 42G free.
- Builds go through `b10x-build-slot`; a timed GPU run gets the GPU alone.

## Units

| # | story | branch | tree id | stage |
|---|---|---|---|---|
| 1 | m1-vulkan-reduce-matmul | impl/m1-vulkan-reduce-matmul | modelspec-impl-reduce | merged a3652af |
| 2 | m1-checkpoint-cpu | impl/m1-checkpoint-cpu | modelspec-impl-ckpt | merged 1f57b4b |

Integration: `wave/m1-c`, tree `modelspec-wave-m1c`, from `main` `23adba4`.
Scratch: `~/.cache/modelspec-wave-m1-c/<story>/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.

## Close

| story | adversary | gate on `1f57b4b` | device on the RTX 3090 |
|---|---|---|---|
| m1-vulkan-reduce-matmul | pass 1 CONFIRMED, 2 findings | `task ci` 1182 passed, 0 failed, exit 0 | class tolerance; 39 + 292 cases run twice, 0 outside tolerance, 0 validation messages |
| m1-checkpoint-cpu | pass 1 NEEDS-CHANGE, 6 findings, fixed; pass 2 CONFIRMED, 5 notes | `task ci` 1182 passed, 0 failed, exit 0 | none (CPU) |

Merge conflicts in `CHANGELOG.md`, the `DESIGN.md` revision line and the generated observatory
schema were resolved at `1f57b4b`; the schema was regenerated and `--check` is current. Both
stories are implemented.
