---
format: aep.planning-md/3
id: review-result:adversary-m1-checkpoint-cpu-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:m1-checkpoint-cpu
relations:
- reviews: story:m1-checkpoint-cpu
revision: 1
---
# Adversary pass 1, story:m1-checkpoint-cpu

Verdict: NEEDS-CHANGE. Target: uncommitted tree of impl/m1-checkpoint-cpu on base 23adba4.
Added tests: crates/modelspec-runtime/tests/adversary_checkpoint.rs (20 cases),
crates/modelspec-cli/tests/adversary_checkpoint_cpu.rs (8 cases). Executed 816 -> 844, red 4.

Red cases:

- a_shape_of_fifty_thousand_dimensions_stays_within_the_allocation_bound: largest allocation
  524288 bytes > file size 100688 + 4096.
- twenty_thousand_records_stay_within_the_allocation_bound: 2883584 bytes > 1180624 + 4096.
- two_checkpoints_that_differ_only_in_their_optimizer_have_different_ids: equal ids
  627cc89e-6ee7-8eb5-bae6-7496d1aeeab8, also reached through train --steps 0 with lr 0.1 and 0.2.
- a_checkpoint_id_that_is_not_a_uuid_is_001: accepted, expected E-CKPT-001.

Mutation sweep: 18 of 22 hand mutants of checkpoint.rs survived the implementor's suite; the
adversary cases kill 15 of them.

Coordinator decisions for fix round 1: bound every manifest allocation (F1); the id covers the
whole manifest except itself (F2); a non-Uuid id is E-CKPT-001 (F4); the spec states that a
ParameterRecord is identified by its path on disk (F5); gap bytes zero, file ends at the last
tensor, payload_hash required (F6).

```findings
[
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":504,"category":"acceptance","severity":"warning","verdict":"NEEDS-CHANGE","origin":"introduced","message":"Manifest JSON parsing grows a Vec<u64> shape or Vec<Record> to 5.2x and 2.4x the file size, breaking the reader's allocation bound and the acceptance headline."},
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":253,"category":"property","severity":"warning","verdict":"NEEDS-CHANGE","origin":"introduced","message":"checkpoint_id omits optimizer and rng, so two different checkpoints (lr 0.1 vs 0.2 at step 0, reached via the CLI) share one identity."},
{"file":"crates/modelspec-runtime/tests/checkpoint_adversary.rs","line":1,"category":"mutant","severity":"warning","verdict":"CONFIRMED","origin":"introduced","message":"18 of 22 hand mutants of checkpoint.rs, including dropped magic, version, overlap, duplicate-path and schema-hash checks, survive the implementor's suite."},
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":194,"category":"contract-drift","severity":"note","verdict":"INFEASIBLE","origin":"introduced","message":"The reader accepts a checkpoint_id that is not a Uuid, which the spec types it as; only a crafted file reaches this."},
{"file":"spec/domains/checkpoint.yaml","line":140,"category":"contract-drift","severity":"note","verdict":"NEEDS-CHANGE","origin":"introduced","message":"The spec gives ParameterRecord a parameter_record_id identity and a checkpoint_id link that the manifest neither writes nor accepts."},
{"file":"crates/modelspec-runtime/src/train/checkpoint.rs","line":571,"category":"contract-drift","severity":"note","verdict":"INFEASIBLE","origin":"introduced","message":"The reader enforces neither the zero gap bytes nor the file ending at the last tensor that DESIGN.md 6.2 states, and verifies no payload when payload_hash is absent."}
]
```
