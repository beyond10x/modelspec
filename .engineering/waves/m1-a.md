# Wave m1-a

Serves `epic:m1-vulkan-execution` and, through it, `epic:pytorch-10x-mnist` (GPU lane).

## Selection

- Units: `story:m1-kernel-plan`, `story:m1-mlp-example`, `story:m1-inspect-device`. N is 3.
- The four stories chef named (`m1-inspect-device`, `m1-vulkan-elementwise`,
  `m1-vulkan-reduce-matmul`, `m1-vulkan-train`) form a chain with three prerequisites the store
  declares: `m1-kernel-plan` and `m1-mlp-example` (both depend only on implemented work), and
  `m1-checkpoint-cpu` (depends on `m1-vulkan-elementwise`). The waves are:
  - m1-a: kernel-plan, mlp-example, inspect-device;
  - m1-b: vulkan-elementwise;
  - m1-c: vulkan-reduce-matmul, checkpoint-cpu;
  - m1-d: vulkan-train.
- `m1-inspect-device` declares `depends_on` `m1-kernel-plan` and `m1-mlp-example`. Its code
  surfaces do not overlap theirs beyond `modelspec-core/src/lib.rs`, `DESIGN.md`, `CHANGELOG.md`,
  `AGENTS.md` and generated reference pages; those conflicts are resolved at the integration
  merges, so it runs in this wave.
- `story:bert-train-data-binding` is not selected: it depends on `story:bert-mlm-batches`, which
  depends on `bert-tokenizer` and `bert-adamw-warmup`, none of them implemented.
- Scope: every unit carries a typed scope and a `## Scope` section from planning; no new scoper
  pass. Implementors confirm each inferred line.
- Added 2026-10-22, after opening: `story:train-data-dense-tensors`, drafted on the coordinating
  session's instruction so the MNIST lane does not wait on the BERT chain (the split is recorded
  in the story's Context). It shares no source file with units 1-3 except `DESIGN.md` and
  `CHANGELOG.md`.

## Approval

`approval-record:m1-waves-and-device-runs`: the operator's go, relayed by b10x-chef-orchestrator
on 2026-10-22, covering the M1 waves, device runs on the RTX 3090 for the M1 stories and the
MNIST benchmark, the unit and merge commits, and landing on `main` through PRs.

## Pre-flight

- `df -h /`: 85G free.
- `b10x-build-slot`: 3 slots, 20G floor.
- Vulkan: `/usr/lib/libvulkan.so.1.4.357`, `vulkaninfo`, `spirv-val` present; RTX 3090 idle at
  237 MiB used (`nvidia-smi`).

## Units

| # | story | branch | tree id | stage |
|---|---|---|---|---|
| 1 | m1-kernel-plan | impl/m1-kernel-plan | modelspec-impl-kplan | dispatched |
| 2 | m1-mlp-example | impl/m1-mlp-example | modelspec-impl-mlp | dispatched |
| 3 | m1-inspect-device | impl/m1-inspect-device | modelspec-impl-inspect | dispatched |
| 4 | train-data-dense-tensors | impl/train-data-dense-tensors | modelspec-impl-dense | dispatched |

Integration: `wave/m1-a`, tree `modelspec-wave-m1a`, from `main` `a5f2b06` (release 0.3.0).
Scratch per unit: `~/.cache/modelspec-wave-m1-a/<story>/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`.
