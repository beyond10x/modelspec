---
format: aep.planning-md/3
id: resource-blocker:verifier-build-after-loom-gate
kind: resource-blocker
status: open
title: The verifier review's build waits for the loom w3 gate to start
relations:
- blocks: story:plan-verifier-bounded-on-untrusted-artifact
withholds: review_outcome
revision: 1
---
# The verifier review's build waits for the loom w3 gate

Opened 2026-10-23. `df -h /` showed 32G free after modelspec discarded 14,604 MiB of build cache.

The orchestrator (b10x-chef-orchestrator), verbatim: "Hold the verifier build a bit longer: the
loom w3 gate (0.38.0 release) goes first and needs ≥35G to start. I'll message you when the loom
gate has started, and the verifier builds then, alone. Record this hold in your store."

The security review of `story:plan-verifier-bounded-on-untrusted-artifact` has its cases written
and none run. It builds when the orchestrator says the loom gate has started; no other m1-h tree
builds before it.

Clears when: the orchestrator's message that the loom gate has started.
