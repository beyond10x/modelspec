---
format: aep.planning-md/3
id: epic:transformer-block-within-3x
kind: epic
status: draft
title: The transformer block within 3× of PyTorch eager
relations:
- serves: vision:modelspec
- derived_from: initiative:transformer-within-3x-of-pytorch
- depends_on: epic:transformer-vs-pytorch-cuda
revision: 1
---
# The transformer block within 3× of PyTorch eager

## Outcome

`examples/transformer_block.mspec` (the two BERT-tiny encoder layers, hidden 128, 2 heads,
feed-forward 512, sequences of 128) trains on the RTX 3090 through Vulkan at batch 8 with a
median step time of at most 3.0× PyTorch eager's on the same device and driver, measured the same
day by `cargo bench -p modelspec-runtime --features vulkan --bench transformer_block` and
`opponents/transformer_block/transformer_block.py`, both rows in `benchmarks/runs.yaml` and on the
Benchmarks page.

Starting point (`verification-report:transformer-block-benchmark-2026-10-27`, release 0.21.0):
93,868.4 µs against 3,937.2 µs at batch 8 (23.8×); 641,794.5 against 12,091.1 at batch 128
(53.1×); 29.0 GFLOP/s against 690.3 on the matmul-only count. The MLP sweep reaches 15,635
GFLOP/s at batch 8192, hidden 4096 (release 0.20.0), so the matmul kernel at large sizes is not
the gap.

## Stories, in order

1. **Profile the block**: `train --backend vulkan --profile` on the example at batch 8, 32 and
   128 on a committed tree, every dispatch site counted (`uncounted_sites 0`), the sites ranked
   by µs per step, the top sites named with their FLOP, bytes and achieved throughput, in a
   `verification-report`; the story's acceptance is that report and the next stories drafted
   from its ranking, one per site or per cause (dispatch count, barrier count, a kernel's
   throughput, host waits), largest first.
2. One story per measured site, drafted after story 1. Candidates the drafts already hold, to be
   confirmed or dropped by the profile: `story:vulkan-train-compiles-once`,
   `story:vulkan-pipeline-cache`, `story:vulkan-buffer-liveness-suballocation`,
   `story:fast-math-cooperative-matrix`, `story:weight-grad-matmul-structural`,
   `story:vulkan-loss-slots-written-by-kernel`, `story:vulkan-scalar-operand-in-reduce-and-loss`.
3. A closing story: the timed runs of the final tree at batch 8, 32 and 128, the report with the
   ratio, the Benchmarks page and the pitch deck updated.

## Measurement rules

Those of `epic:transformer-vs-pytorch-cuda` § Measurement rules: the training loop only, median of
at least 3 timed runs of at least 5 s, the GPU alone, clocks never locked, device runs the
coordinator's. Every story's figure is measured on its committed tree with the unit 22 bench; an
implementor's smoke run (`--min-seconds 1`, dirty tree) is never a figure. Numerics stay within
the DESIGN.md § 9 run tolerance of the CPU reference at every batch, checked by the 10-step
agreement test and the opponent's `--compare-step`.

## Out of scope

CUDA or any vendor API; `fast_math` by default; a larger model than BERT-tiny; the observatory and
the IR gallery; changing the FLOP rule.

## Dependencies

`epic:transformer-vs-pytorch-cuda` (the bench, the opponent, the figures), `epic:m1-vulkan-execution`.
