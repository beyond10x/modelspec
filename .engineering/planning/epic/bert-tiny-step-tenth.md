---
format: aep.planning-md/3
id: epic:bert-tiny-step-tenth
kind: epic
status: proposed
title: 'BERT-tiny on Vulkan: a training run at a tenth of its 2026-10-28 wall time'
summary: 432 ms per step (half held-out evaluation, half four badly parallelised kernels) down to at most 43 ms, evaluation included, results bit-identical where the stories say so
relations:
- serves: vision:modelspec
revision: 2
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T20:18:18Z", actor: "human:timo", revision: 2}
---
# BERT-tiny on Vulkan: a training run at a tenth of its 2026-10-28 wall time

## Request

The operator, 2026-10-28, after the timed runs of story:bert-tiny-vs-pytorch-cuda: "unless you
bring model spec time per run down to say a tenth, there is no point in even attempting a new
run." Target: at most 43 ms of wall time per step, evaluation included, on
`examples/bert_tiny.mspec` at batch 32 × 128 on the RTX 3090 (432 ms per step on 2026-10-28,
verification-report:bert-tiny-vs-pytorch-2026-10-28). No new timed 10,000-step run before the
target is met on short runs.

## Diagnosis (2026-10-28, at 3b9869f)

From the timed-run files: training clock 2225.8 s, evaluation 2093.1 s, wall 4330 s per run.

| part | ms per step |
|---|---|
| training step | 222.6 |
| held-out evaluation (101 evaluations × 68 batches) | 209.3 |
| outside both clocks | 1.1 |

GPU timestamps over 200 steps: 256.8 ms of GPU time per profiled step against about 2 ms of host
work (589 dispatches, 412 barriers, 1 submit, 1 fence). PyTorch eager: 300 kernels, 6.5 ms CUDA
time, 12.4 ms wall per step; 0.21 s per evaluation.

| hypothesis | probe | before → after | verdict |
|---|---|---|---|
| evaluation is host-bound (134 MB of logits copied and log-softmaxed per batch) | skip weight-0 rows, drop the loss-sum dispatch, read in place | 21.7 s → 3.4 s per evaluation | confirmed |
| AttentionGradK/V recompute each query's softmax max, sum and row dot in every key invocation | the three values set to constants (timing only) | GradK 120.3 → 5.1 ms, GradV 60.3 → 4.9 ms | confirmed |
| the CrossEntropy loss sum runs as one workgroup over 64 serial blocks | one block only (timing only) | 36.3 → 0.6 ms | confirmed |
| host overhead per step | steady step time minus GPU total | about 2 ms | refuted |
| bookkeeping (statistics, events, stdout) | event sizes, time outside the clocks | 1.1 ms | refuted |

## Plan

| order | story | expected gain (ms per step) | step after |
|---|---|---|---|
| 0 | done on diag/bert-step-bottleneck (52c2fd4): evaluation without the loss sum, weight-0 rows skipped bit-exactly, mapped read | evaluation 209 → about 34 | about 258 |
| 1 | story:vulkan-attention-softmax-stats-reused | about −149 | about 109 |
| 2 | story:vulkan-cross-entropy-sum-per-row | about −31 | about 78 |
| 3 | story:vulkan-eval-metrics-on-device | evaluation about 34 → about 10 | about 54 |
| 4 | story:vulkan-embedding-grad-per-element | about −9 | about 45 |
| 5 | story:vulkan-attention-workgroup-per-head | about −14 (estimated, not probed) | about 31 plus evaluation about 4 |

Gains are profiled shares scaled to unprofiled time (`--profile` adds about 35 ms per step).
Every story measures its own before and after on a 200-step run and a 20-step run with
evaluation, and keeps results bit-identical to the current Vulkan run unless its own text says
otherwise (DESIGN.md § 9).

Decision D128 (coordinator, 2026-10-28): story:vulkan-eval-metrics-on-device computes the
held-out NLL with the device's exp and ln; the held-out loss on Vulkan is then held within the
run tolerance of DESIGN.md § 9, not bit-identical to the host tally, as the Vulkan training loss
already is.
