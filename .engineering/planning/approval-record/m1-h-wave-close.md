---
format: aep.planning-md/3
id: approval-record:m1-h-wave-close
kind: approval-record
status: draft
title: Wave m1-h decisions taken at close
relations:
- serves: vision:modelspec
revision: 1
---
# Wave m1-h decisions taken at close

- D1: the CI ESS pin moved to 0.68.0, with its sha256, in `.github/workflows/check.yml` (904a64e),
  because `spec/ess-inputs.yaml` requires 0.68.0 and CI refused the mismatch.
- D2: the live observatory UI test starts its browser before `train --serve` (6fd6c30), because in
  CI, with 6 browser tests in parallel, Chrome started after the 50000-step run had finished.
- D3: the 6 merged unit branches were deleted through the bot; `design/targets` is kept as a paused
  draft (approval-record:design-targets-paused).
- D4: the settlements of unit 5 are in approval-record:parameter-statistics-settlements.
