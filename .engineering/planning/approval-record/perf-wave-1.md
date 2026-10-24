---
format: aep.planning-md/3
id: approval-record:perf-wave-1
kind: approval-record
status: accepted
title: 'Performance wave 1: approval and scope'
tags:
- operator-word
relations:
- serves: vision:modelspec
- decides: story:cpu-prepared-training-step
- decides: story:dense-step-reuses-batch-buffers
- decides: story:vulkan-step-counters
- decides: story:cpu-matmul-output-skips-zero-fill
- decides: story:parser-span-lookup-not-linear-in-column
revision: 2
transitions:
- {from: "draft", to: "accepted", at: "2026-10-24T08:09:08Z", actor: "human:timo", revision: 2}
---
# Performance wave 1: approval and scope

Taken by the modelspec session on 2026-10-24 on the operator's words of the same day, "I accept
all you want, you act on behalf of me", and the goal he set, "performance issues addressed from
review and new version cut". The orchestrator's grant (exception 6) covers the wave as a class-C
item.

## What the approval covers

The unit commits on the five `impl/<story>` branches through the bot, the adversary's cases on
those branches, the merges into `wave/perf-1`, the closing store commit, the merge of
`wave/perf-1` into `main` by fast-forward after its checks, and the release that follows the
performance waves. No force-push, no history rewrite, no registry publishing.

## Units

Selected by `aep plan artifact waves --status draft --format json` at the review-drafts head
(28 draft stories of the two code reviews, 0 cycles): the five performance stories with no
pairwise collision and no open `depends_on`.

- Unit 1: story:cpu-prepared-training-step (verify once per run, free after last consumer, moved
  outputs; `verify.rs`, `cpu/interpreter.rs`, `train/optimizer.rs`).
- Unit 2: story:dense-step-reuses-batch-buffers (`train/data.rs`, `train/runner.rs`).
- Unit 3: story:vulkan-step-counters (`--stats` counters the fusion and liveness stories read;
  `vulkan/{memory,profile,runner,sync}.rs`, `cli/train.rs`, DESIGN.md § 8, the guide, the CLI
  reference).
- Unit 4: story:cpu-matmul-output-skips-zero-fill (`cpu/matmul.rs`).
- Unit 5: story:parser-span-lookup-not-linear-in-column (`syntax/parser.rs`).

## Left out of this wave, and why

- story:schedule-index-tables collides with unit 1 on `crates/modelspec-core/src/ir/verify.rs`.
- story:cpu-finite-check-updated-params-only collides with unit 2 on `train/runner.rs`.
- story:vulkan-train-compiles-once collides with unit 3 on `crates/modelspec-cli/src/train.rs`.
- story:profile-begin-stamp-after-barrier, story:vulkan-pipeline-cache and
  story:vulkan-loss-slots-written-by-kernel collide with unit 3 on `vulkan/runner.rs`.
- story:fuse-elementwise-chains-and-gradient-sums and
  story:vulkan-buffer-liveness-suballocation depend on unit 3.
- story:matmul-shape-aware-tiling depends on story:profile-begin-stamp-after-barrier;
  story:fast-math-mode depends on it and on story:vulkan-loss-slots-written-by-kernel.

The planned order after this wave: wave 2 (schedule-index-tables, profile-begin-stamp,
cpu-finite-check, vulkan-train-compiles-once), wave 3 (fuse, vulkan-pipeline-cache), then
matmul-shape-aware-tiling, vulkan-buffer-liveness-suballocation, vulkan-loss-slots-written-by-
kernel and fast-math-mode, which pairwise share `schedule.rs`, `emit.rs` or `vulkan/runner.rs`.
