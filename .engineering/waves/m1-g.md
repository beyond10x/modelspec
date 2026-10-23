# Wave m1-g

Five stories, merged into one integration branch and delivered as one pull request, as the operator
asked on 2026-10-23.

## Selection

| # | story | lane | device run |
|---|---|---|---|
| 1 | `story:m1-f-weight-grad-matmul-tiled` | GPU, `epic:pytorch-10x-mnist` | yes |
| 2 | `story:train-refuses-non-finite-loss` | CPU training | no |
| 3 | `story:m1-compile-artifact` | CLI, `epic:m1-vulkan-execution` | no |
| 4 | `story:m1-vulkan-checkpoint-infer` | GPU, `epic:m1-vulkan-execution` | yes |
| 5 | `story:observatory-run-recording-replay` | `epic:training-observatory` | no |

- `aep plan artifact waves --status active` puts each story in its own wave: every pair shares
  `CHANGELOG.md`, and most share `DESIGN.md` or a CLI file. The units run in the order above.
- Unit 3 lands before unit 4, so the status row "Checkpoints and the compile and infer commands"
  (`crates/modelspec-docs/src/status.rs`) is unit 3's; unit 4 adds the device half.
- Not selected: `story:bert-corpus-splits` and `story:bert-embedding-lookup` depend on the draft
  `story:observatory-saliency-gradient-x-input`; corpus-splits also waits on the operator's corpus
  and licence decision.

## Approval

- `approval-record:m1-g-wave`: the operator's request of 2026-10-23.
- `approval-record:m1-waves-and-device-runs`: device runs on the RTX 3090.
- `approval-record:m1-tolerance-numerics-class`: per-operator tolerances in every GPU test.

## Store changes before the units

- Four stories moved draft, proposed, active; unit 1 was already active.
- `story:m1-vulkan-checkpoint-infer`: its `depends_on` edge to its parent epic (draft) became
  `decomposes`; `DESIGN.md`, the runner, the two existing tests that change and three inferred files
  were added to its scope, and the `vulkan/` directory entry was replaced by `vulkan/runner.rs`.
- `story:observatory-run-recording-replay` and `story:train-refuses-non-finite-loss`: scope
  recorded from the story scopers.

## Settled by the coordinator

- Unit 1: acceptance 5 (at most 31.4 µs per step) is met by release 0.5.0 at 27.282 µs, and
  acceptance 1's `k11` figures predate unit 1 of wave m1-f. The story's acceptance is re-based on a
  `k11` profile of `main` `7c45c1a` before any code (see Units).
- Unit 5: the marker `UNMAPPED (recorded file)` of `spec/domains/observatory.yaml` is settled by the
  wave m1-g coordinator, as the wave observatory-1 coordinator settled the earlier markers there.

## Starting point

`main` `7c45c1a` (release 0.5.0). ADR-007 method: 27.282 µs per step, median of 5 runs of 144,000
steps (27.267 to 27.435), mean P0 1957 to 1965 MHz. 1,800-step stdout sha256 `b9bb028a…6904`.

## Pre-flight

- `df -h /`: 62G free. Builds go through `b10x-build-slot`, one at a time, paused under 25G free.
- A timed GPU run gets the GPU alone.

## Units

| # | story | branch | tree id | stage |
|---|---|---|---|---|
| 1 | m1-f-weight-grad-matmul-tiled | impl/m1-f-weight-grad-matmul-tiled | modelspec-impl-m1g-wgrad | merged 72bcdef; k11 4.58 µs, miss rule, gap story m1-f-weight-grad-matmul-columns |
| 2 | train-refuses-non-finite-loss | impl/train-refuses-non-finite-loss | modelspec-impl-m1g-nonfinite | merged 5c03d9a; doc contract fixed, Vulkan check filed as train-vulkan-refuses-non-finite-loss |
| 3 | m1-compile-artifact | impl/m1-compile-artifact | modelspec-impl-m1g-compile | merged dc631b0; 4 reader fixes, verifier bound filed as plan-verifier-bounded-on-untrusted-artifact |
| 4 | m1-vulkan-checkpoint-infer | impl/m1-vulkan-checkpoint-infer | modelspec-impl-m1g-ckpt | implementing |
| 5 | observatory-run-recording-replay | impl/observatory-run-recording-replay | modelspec-impl-m1g-replay | merged b55deb1; 3 adversary findings fixed in 2449dda |

Integration: `wave/m1-g`, tree `modelspec-wave-m1g`, from `main` `7c45c1a`.
Scratch: `~/.cache/modelspec-wave-m1-g/<story>/scratch/`.

Dispatch: `aep:implementor`, then `aep:adversary`, one unit at a time.
