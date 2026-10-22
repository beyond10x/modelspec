# Wave m1-b

Serves `epic:m1-vulkan-execution` and, through it, `epic:pytorch-10x-mnist` (GPU lane).

## Selection

- Unit: `story:m1-vulkan-elementwise`. N is 1.
- Its `depends_on` stories (`m1-inspect-device`, `m1-kernel-plan`, `m1-mlp-example`) are
  implemented on `main` `6937dac` (wave m1-a). The scope notes on `story:cpu-matmul-row-order` and
  `story:observatory-live-event-stream` are satisfied: both are implemented on `main`.
- Next waves: m1-c (`m1-vulkan-reduce-matmul`, `m1-checkpoint-cpu`), m1-d (`m1-vulkan-train`).

## Approval

- `approval-record:m1-waves-and-device-runs`: the M1 waves and device runs on the RTX 3090.
- `approval-record:m1-b-start`: land wave m1-a and start this wave without a further go.
- `approval-record:m1-tolerance-numerics-class`: every GPU test names its tolerance per operator.

## Pre-flight

- `df -h /`: 47G free.
- Builds go through `b10x-build-slot`; a timed GPU run gets the GPU alone.

## Units

| # | story | branch | tree id | stage |
|---|---|---|---|---|
| 1 | m1-vulkan-elementwise | impl/m1-vulkan-elementwise | modelspec-impl-elementwise | dispatched |

Integration: `wave/m1-b`, tree `modelspec-wave-m1b`, from `main` `6937dac`.
Scratch: `~/.cache/modelspec-wave-m1-b/m1-vulkan-elementwise/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.
