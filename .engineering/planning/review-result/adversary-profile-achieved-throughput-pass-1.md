---
format: aep.planning-md/3
id: review-result:adversary-profile-achieved-throughput-pass-1
kind: review-result
status: active
title: 'Adversary pass 1: no wrong count, three untested guards'
relations:
- reviews: story:profile-achieved-throughput
revision: 1
---
# Adversary pass 1 on story:profile-achieved-throughput

Tree at 7e8cbde on base 4c80e34. Verdict: no wrong count; one warning, two notes, each an untested
guard. The three cases that kill them were committed in 939a698.

Suite: 1394 before, 1398 after the two adversary files; `cargo fmt --check`, both clippy runs and
`modelspec-docs generate --check` exit 0. No `--ignored` test was run.

Red against scratch mutants:
- M1, `Add` and `Relu` dropped from the rule at kernel.rs:399: `left: [Uncounted] right: [Counted { flop: 32, bytes: 384 }]`.
- M2, `checked_*` replaced by `wrapping_*`: `left: Some(0) right: None`.
- M3, the `us > 0.0` filter removed at profile.rs:210: `gflop_per_s inf gb_per_s inf`.

Attacked without a break: fused ReLU with a pre-activation write (565504 bytes, 0 FLOP for the
epilogue), two-stage reductions and loss_sum, D2 (module ids in both goldens unchanged), units
(decimal GFLOP/s and GB/s), every number in the guide and DESIGN.md.

```findings
- file: crates/modelspec-core/src/ir/kernel.rs
  line: 399
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: no case counted an Add or an unfused Relu, so dropping both from the rule printed uncounted with every case green
- file: crates/modelspec-core/src/ir/kernel.rs
  line: 396
  category: contract-drift
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: no case reached the u64 overflow path, and a wrapping mutant printed flop 0; compiled plans cannot reach it
- file: crates/modelspec-runtime/src/vulkan/profile.rs
  line: 210
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: the documented unavailable-for-time-0 output was untested and its removal printed inf and NaN
```
