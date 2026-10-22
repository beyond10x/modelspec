---
format: aep.planning-md/3
id: approval-record:m1-g-wave
kind: approval-record
status: draft
title: 'Wave m1-g approved: five stories in one integration pull request'
relations:
- serves: vision:modelspec
revision: 1
---
# Wave m1-g approved: five stories in one integration pull request

- Requested by the operator on 2026-10-23: "select the next wave of 5 work items - implement all of
  them, merge all into this integration PR - submit one 1 PR". The release of 0.5.0 came first, as
  the operator asked ("commit and cut a new version first").
- Units, in this order, one implementor at a time, one build at a time through `b10x-build-slot`:
  1. `story:m1-f-weight-grad-matmul-tiled` (active), its acceptance re-based on a `k11` profile of
     `main` `7c45c1a` before any code.
  2. `story:train-refuses-non-finite-loss`.
  3. `story:m1-compile-artifact`. It lands before unit 4, so the status row "Checkpoints and the
     compile and infer commands" is owned by this story and unit 4 only adds the device half.
  4. `story:m1-vulkan-checkpoint-infer`.
  5. `story:observatory-run-recording-replay`. The recorded-file marker of
     `spec/domains/observatory.yaml` is settled by the wave m1-g coordinator, as the wave
     observatory-1 coordinator settled the earlier markers in that file.
- Substituted for `story:bert-corpus-splits` and `story:bert-embedding-lookup`, which depend on the
  draft `story:observatory-saliency-gradient-x-input`; corpus-splits also waits on the operator's
  corpus and licence decision.
- Commits this grants: one per unit, the merges into `wave/m1-g`, the store commits and the
  fast-forward of `main` once the one pull request is green, all through the bot. No tag or release.
- Device runs on the RTX 3090 stay under `approval-record:m1-waves-and-device-runs`; a timed run
  gets the GPU alone.
