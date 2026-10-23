---
format: aep.planning-md/3
id: review-result:adversary-train-vulkan-refuses-non-finite-loss-pass-1
kind: review-result
status: active
title: 'Adversary pass 1: a diverged Vulkan run printed no --stats counters'
relations:
- reviews: story:train-vulkan-refuses-non-finite-loss
revision: 1
---
# Adversary pass 1 on story:train-vulkan-refuses-non-finite-loss

Tree at a2a2944. Verdict: one warning, no blocker.

Added `crates/modelspec-cli/tests/train_vulkan_non_finite_adversary.rs`: 7 cases, each run on CPU and
Vulkan, comparing exit status, stdout, stderr and checkpoint file names. 6 green, 1 red.

Attacked without a break: step and value named at checkpoint window edges (every 1, 2, 7, 8, 9),
the final loss, chunk boundaries 255/256/257, a divergence at step 372, `--resume` from step 256,
`--serve` stream bodies in 5 configurations, host waits (1 + number of checkpoints), MNIST at lr
10, 100, 1e30 and 1e38.

Red: `a_diverged_run_with_stats_still_prints_its_counters` (lr=100 with `--stats`): no `host waits:
1` line after the E-TRAIN-001 diagnostic. At base 24e84d0 the same run printed it.

```findings
- file: crates/modelspec-cli/src/train.rs
  line: 603
  category: contract-drift
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: a diverged --backend vulkan run with --stats or --profile prints no counters, against DESIGN.md § 8 "--stats prints the run's counters on stderr after it", where base printed host waits 1 for the same run
```
