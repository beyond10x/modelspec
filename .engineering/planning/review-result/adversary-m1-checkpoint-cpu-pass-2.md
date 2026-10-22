---
format: aep.planning-md/3
id: review-result:adversary-m1-checkpoint-cpu-pass-2
kind: review-result
status: active
title: Adversary pass 2, story:m1-checkpoint-cpu
relations:
- reviews: story:m1-checkpoint-cpu
revision: 1
---
# Adversary pass 2, story:m1-checkpoint-cpu

Verdict: CONFIRMED, nothing red. Target: fix round 1 on impl/m1-checkpoint-cpu, base 23adba4.
Adversary tests: runtime 28 -> 30 (two MAX_RANK edge cases), cli 8; red 0.
`b10x-build-slot task ci` exit 0.

Every fix of round 1 held against its attack: deep nesting and repeated keys in the manifest
(F1), the record bound at its edge (F1), id determinism across resume and chained segments (F2),
a non-Uuid id (F4), reader and writer round trips with 1 and 10 tensors, non-zero gaps and
trailing bytes, an absent payload_hash (F6), `ess specify validate` on the spec change.

Mutation sweep on the round-1 code: M22b, M23, M26, M30, M27, M28 survive the implementor's
suite and are killed by the adversary suite; M09, M18, M20, M21, M29 survive both and are
equivalent or unreachable (notes below).

```findings
[
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":849,"category":"mutant","severity":"note","verdict":"INFEASIBLE","origin":"introduced","message":"M09 (no zero-dimension check) survives both suites because a later check refuses the file first."},
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":852,"category":"equivalent","severity":"note","verdict":"INFEASIBLE","origin":"introduced","message":"M18 (length 0 allowed) looks equivalent, since length must equal the element count times 4 and no dimension may be 0."},
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":503,"category":"equivalent","severity":"note","verdict":"INFEASIBLE","origin":"introduced","message":"M20 (no dtype check in parameters_for) is unreachable while the only dtype is F32."},
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":116,"category":"equivalent","severity":"note","verdict":"INFEASIBLE","origin":"introduced","message":"M21 (learning_rate.is_finite skipped) is unreachable because JSON cannot encode NaN or infinity."},
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":402,"category":"contract-drift","severity":"note","verdict":"INFEASIBLE","origin":"introduced","message":"encode() does not refuse a parameter of more than MAX_RANK dimensions; only write() does, so encode can produce bytes that read() refuses with E-CKPT-001."}
]
```
