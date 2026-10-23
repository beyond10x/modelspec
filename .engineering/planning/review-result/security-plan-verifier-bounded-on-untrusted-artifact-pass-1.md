---
format: aep.planning-md/3
id: review-result:security-plan-verifier-bounded-on-untrusted-artifact-pass-1
kind: review-result
status: active
title: 'Security pass 1: manifest parsed before its size check, DESIGN.md timing claim wrong'
relations:
- reviews: story:plan-verifier-bounded-on-untrusted-artifact
revision: 1
---
# Security review pass 1 on story:plan-verifier-bounded-on-untrusted-artifact

Tree at c3f9aa0 on base 24e84d0. Verdict: two warnings, one note, no blocker.

Added `crates/modelspec-core/tests/adversary_plan_verifier_bounds.rs` (4 cases: A, B, B control, E).
Suite with the file: 332 run, 329 pass, 3 fail (A, B, E); without it all 328 existing tests pass.
Clippy `-p modelspec-core --all-targets -D warnings` exit 0.

Attacked without a break: 20,000 random plans and 6,000 back-to-back runs differentially against
the base verifier (no disagreement); the 300,000-step chain unpinned (4.07 to 4.22 s, 463 MB) and
with a CPU hog on its CPU (8.30 s); duplicate dependencies and kernel ids; overflow in
`check_plan_sizes`.

Red:
- A: a sealed 67,108,863-byte manifest with 33,553,622 dependencies takes 15.976 s and
  1,381,504 KiB before E-ARTIFACT-003, because `read_manifest` parses into `serde_json::Value`
  before `check_plan_sizes`.
- B: an unsynced Copy pair at MAX_PLAN_VIEWS verifies in 2.96 to 3.71 s; DESIGN.md:469 claims
  under 0.2 s. `conflict()` compares every read with every write.
- E: `verify_recorded_back_to_back` keeps bit sets over n times the steps; only tests call it.

```findings
- file: crates/modelspec-core/src/artifact.rs
  line: 582
  category: bounds
  severity: warning
  verdict: CONFIRMED
  origin: pre-existing
  message: "read_manifest parses the whole manifest into serde_json::Value before check_plan_sizes; a 67,108,863-byte manifest with 33,553,622 dependencies takes 15.976 s and 1,381,504 KiB before E-ARTIFACT-003, past the 10 s / 1 GiB bound."
- file: DESIGN.md
  line: 469
  category: bounds
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "Claims a 4,096-step plan verifies in under 0.2 s in debug; an unsynced Copy pair at MAX_PLAN_VIEWS takes 2.96-3.71 s because conflict() at crates/modelspec-core/src/ir/verify.rs:1174 is O(reads x writes)."
- file: crates/modelspec-core/src/ir/verify.rs
  line: 930
  category: bounds
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: "verify_recorded_back_to_back memory is unbounded in n; the MNIST plans recorded 4096 times pass 1 GiB in 2.4 s. Only tests call it."
```
