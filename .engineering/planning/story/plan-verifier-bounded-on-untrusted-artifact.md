---
format: aep.planning-md/3
id: story:plan-verifier-bounded-on-untrusted-artifact
kind: story
status: active
title: The plan verifier stays bounded on an untrusted artifact
relations:
- decomposes: epic:m1-vulkan-execution
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/tests/compile_adversary.rs
- confidence: cited
  path: crates/modelspec-core/src/artifact.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: inferred
  path: crates/modelspec-core/tests/adversary_artifact_reader.rs
- confidence: inferred
  path: crates/modelspec-core/tests/adversary_kplan_verifier.rs
- confidence: inferred
  path: crates/modelspec-core/tests/kernel_plan_adversary.rs
- confidence: inferred
  path: spec/domains/artifact.yaml
revision: 12
transitions:
- {from: "draft", to: "proposed", at: "2026-10-23T08:16:29Z", actor: "human:timo", revision: 2}
- {from: "proposed", to: "active", at: "2026-10-23T08:16:29Z", actor: "human:timo", revision: 3}
---
# The plan verifier stays bounded on an untrusted artifact

## Context

The wave m1-g adversary of `story:m1-compile-artifact` read `crates/modelspec-core/src/ir/verify.rs:1214`:
the verifier keeps a dependency closure set per step, so its memory grows with the square of the
number of steps. `artifact::read` now feeds it plans from disk, up to a 64 MiB manifest. A re-sealed
manifest holding a chain of about 300,000 steps would need more memory than the machine has. This
is inferred from the code; no test ran it.

## What it changes

- The verifier's reachability check uses memory linear in the plan (for example a topological order
  with per-buffer last-writer tracking), or the reader refuses a plan above a stated step limit with
  E-ARTIFACT-003 before the verifier runs.
- Every plan the compiler emits today still verifies, with the same result.

## Scope

- `crates/modelspec-core/src/ir/verify.rs`, `crates/modelspec-core/src/artifact.rs`, their tests,
  `DESIGN.md` § 7.1 when a limit is added, `CHANGELOG.md`.

## Acceptance

1. A re-sealed artifact with a 300,000-step dependency chain is verified or refused in under 10 s
   and under 1 GiB of resident memory.
2. Every existing verifier and artifact test passes unchanged.
3. `task ci` exits 0.
