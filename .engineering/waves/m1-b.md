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
| 1 | m1-vulkan-elementwise | impl/m1-vulkan-elementwise | modelspec-impl-elementwise | merged `adca91f` (unit `869f26e`) |

Integration: `wave/m1-b`, tree `modelspec-wave-m1b`, from `main` `6937dac`.
Scratch: `~/.cache/modelspec-wave-m1-b/m1-vulkan-elementwise/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.

## Close

- Adversary passes: 2 (6 and 4 red device or suite cases, plus 1 suite gap), all fixed and
  verified green by the coordinator. Recorded as
  `review-result:adversary-m1-vulkan-elementwise-pass-<n>` with a `fixed` review outcome each.
- Coordinator decision in fix round 1: on a device that does not report f32 denorm preservation
  (the RTX 3090), gate 5 compares against the CPU interpreter on inputs with subnormals flushed to
  signed zero (DESIGN.md 0.13). `no_vulkan.rs` also bars `rspirv` and `spirv` from
  `modelspec-core`.
- Gate: `b10x-build-slot task ci` on `adca91f`, 1048 passed, 0 failed, exit 0.
- RTX 3090, validation on, on `adca91f`: 18 device tests passed; 120 cases, 1653800 elements,
  0 outside tolerance, 0 guard words changed, 0 validation messages; the chained case waits once.
- GitHub `check.yml` on PR 14 (`adca91f`): spirv-tools installed, the 4 `spirv_val` tests pass.
- Moved to implemented: `story:m1-vulkan-elementwise`.
- Next: wave m1-c (`m1-vulkan-reduce-matmul`, `m1-checkpoint-cpu`).
