---
format: aep.planning-md/3
id: story:manifest-arrays-bounded-on-untrusted-artifact
kind: story
status: draft
title: Bound the remaining manifest arrays of an untrusted artifact
relations:
- informed_by: review-result:security-plan-verifier-bounded-on-untrusted-artifact-pass-1
revision: 1
---
# Bound the remaining manifest arrays of an untrusted artifact

## Outcome

`modelspec_core::artifact::read` refuses an untrusted `artifact.json` whose parameter shapes, data
shapes, `cache_key.capabilities`, `cache_key.shapes` or `buffers` arrays exceed a declared limit,
in under 10 s and 1 GiB resident in a debug build, with an `E-ARTIFACT-*` diagnostic, before it
holds the array.

## Why

Round 2 of story:plan-verifier-bounded-on-untrusted-artifact bounded the plan arrays (steps,
dependencies, views). A probe in that round measured five other arrays with no limit, each in a
manifest under the 64 MiB cap:

| Array | Measured |
|---|---|
| parameter shape | past 10 s |
| data shape | past 10 s |
| cache_key.capabilities | past 10 s |
| cache_key.shapes | past 1 GiB at 8.5 s |
| buffers | 8.3 s, 518 MiB |

DESIGN.md § 7.1 records them as unbounded.

## Acceptance

1. Each of the five arrays has a named limit constant, chosen above the largest value any
   example model in the repository produces, and stated in DESIGN.md § 7.1.
2. A test per array builds a sealed manifest past its limit and asserts refusal in under 10 s and
   1 GiB resident, run in a child process as `tests/plan_verifier_bounds.rs` does.
3. Every existing artifact and verifier test passes unchanged; `task ci` exits 0.

## Open

The limits are a specification decision: measure the largest real values first.
