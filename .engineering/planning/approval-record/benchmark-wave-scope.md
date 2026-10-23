---
format: aep.planning-md/3
id: approval-record:benchmark-wave-scope
kind: approval-record
status: draft
title: 'Benchmark wave: units and scope decisions'
relations:
- serves: vision:modelspec
revision: 2
---
# Benchmark wave: scope decisions

Taken by the modelspec session on 2026-10-23 under the orchestrator's grant
(approval-record:benchmark-wave-after-m1-h), from the story-scoper reports for
story:profile-achieved-throughput and story:mlp-size-sweep-vs-pytorch-cuda at main 6fd6c30.

## Units

- Unit 1: story:profile-achieved-throughput (`modelspec-core` kernel-plan counts and
  `ir --stage kernel`; `modelspec-runtime/src/vulkan/profile.rs` report lines; DESIGN.md; CLI
  reference).
- Unit 2: story:mlp-size-sweep-vs-pytorch-cuda (`crates/modelspec-runtime/benches/mlp_sweep.rs`,
  its `[[bench]]` entry, the PyTorch opponent under `opponents/`).
- story:benchmark-results-page waits: its acceptance shows figures from two verification reports
  that only device runs produce, and device runs are Timo's.
- Units 1 and 2 run in parallel. Shared file: `crates/modelspec-runtime/src/vulkan/runner.rs`,
  read by unit 2 and changed by unit 1 only if it must; the integration merge settles any conflict.

## Decisions

- D1 Counting rule (one rule for both stories). MatMul: 2·m·k·n FLOP. Elementwise kinds,
  including `sgd_update`: FLOP per output element as the kernel's arithmetic states it.
  Reductions, including `loss_sum` and `reduce_stage_*`: one FLOP per input element.
  `transpose` and `broadcast`: 0 FLOP, bytes counted. Bytes: the sum of the dispatch's read
  views' `size_bytes` plus its write views' `size_bytes` (nominal traffic, not measured DRAM
  traffic). A kind without a rule yields no count and prints `uncounted`; the count type keeps
  that case, and a unit test reaches it.
- D2 Counts are computed from the plan on read. No wire field of the artifact, no artifact hash
  and no module id changes.
- D3 Per-step GFLOP/s is the step's counted FLOP divided by the steady µs per step; the step line
  names how many sites were uncounted.
- D4 The reduction unit test may build a kernel plan in the test; no example file is added for it.
- D5 Sweep data is synthetic (the teacher–student generator); `train --backend vulkan` takes no
  data input (`crates/modelspec-cli/src/train.rs:332`).
- D6 Sweep sizes: the bench formats one source text per point in Rust and compiles it; no
  `.mspec` files are committed for the sweep (the language has integer-literal dimensions only).
- D7 Sweep FLOP per step is the matmul count, 3196·B·H for this model family, and the report says
  it is matmul-only. Under D1 that equals unit 1's matmul total for the same plan.
- D8 Sweep µs per step is host wall time per step, with setup, data generation and the 20-step
  warm-up outside the timed span; the PyTorch side times the same span with a CUDA synchronize
  at both ends. Each side's figure is the median of 3 runs (the epic's rule).
- D9 The PyTorch opponent lives in `opponents/mlp_sweep/` (Python is allowed only under
  `opponents/`, AGENTS.md); it runs eager and `torch.compile`, CUDA, TF32 off.
- D10 Device runs are not run by this session. Each unit ends with the exact command and what it
  measures, sent to the orchestrator for Timo's queue.
- D11 The sweep trains at lr 0.001, not the MNIST example's 0.1: the implementor estimated 0.1 diverges at hidden 1024 and 4096, which would turn those points into refusals, and step time does not depend on the learning rate (adversary pass 1, 2026-10-23). Not measured.
- D12 A crossover verdict "undetermined" is added beside a width and "not reached": a width with no figure before any crossover leaves the answer open instead of deciding it.
