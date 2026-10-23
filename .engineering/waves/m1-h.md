# Wave m1-h

Five stories, merged into one integration branch and delivered as one pull request, approved by the
orchestrator on 2026-10-23 (`approval-record:m1-h-wave`).

## Selection

| # | story | lane | device run |
|---|---|---|---|
| 1 | `story:train-vulkan-refuses-non-finite-loss` | GPU, `epic:pytorch-10x-mnist` | yes |
| 2 | `story:plan-verifier-bounded-on-untrusted-artifact` | core, `epic:m1-vulkan-execution` | no |
| 3 | `story:m1-f-weight-grad-matmul-columns` | GPU, `epic:pytorch-10x-mnist` | yes, timed |
| 4 | `story:observatory-ui-live-loss-chart` | UI, `epic:training-observatory` | no |
| 5 | `story:observatory-parameter-statistics` | `epic:training-observatory` | no |

- Units 1 to 4 run in parallel, each in its own tree; unit 5 depends on unit 4 and starts after it
  merges. Every unit shares `CHANGELOG.md`; units 1, 2 and 3 share `DESIGN.md`; units 2 and 3 share
  `crates/modelspec-core/src/artifact.rs` (inferred for both); units 4 and 5 share
  `crates/modelspec-observatory/` and `spec/domains/observatory.yaml`. The coordinator resolves
  these at merge.
- Scope recorded from the story scopers before dispatch.

## Settled by the coordinator

- Unit 1: no per-step read-back. The device checks the loss slots and the parameters where they are
  already read back (checkpoint stops, the end); `infer --backend vulkan` does not change.

Integration: `wave/m1-h`, tree `modelspec-wave-m1h`, from `main` `24e84d0`.
Scratch: `~/.cache/modelspec-wave-m1-h/<story>/scratch/`.

Dispatch: `aep:implementor` per unit, then `aep:adversary`.

## Units

| # | story | branch | tree | state |
|---|---|---|---|---|
| 1 | train-vulkan-refuses-non-finite-loss | impl/train-vulkan-refuses-non-finite-loss | modelspec-impl-m1h-nonfinite-vk | merged, implemented |
| 2 | plan-verifier-bounded-on-untrusted-artifact | impl/plan-verifier-bounded-on-untrusted-artifact | modelspec-impl-m1h-verifier | merged, implemented |
| 3 | kernel-signature-covers-matmul-tile (re-scoped from m1-f-weight-grad-matmul-columns) | impl/kernel-signature-covers-matmul-tile | modelspec-impl-m1h-wgrad-cols | merged, implemented |
| 4 | observatory-ui-live-loss-chart | impl/observatory-ui-live-loss-chart | modelspec-impl-m1h-ui-chart | merged at c7daf8b, implemented |
| 5 | observatory-parameter-statistics | impl/observatory-parameter-statistics | modelspec-impl-m1h-param-stats | 6272b74, adversary pass 1 |
