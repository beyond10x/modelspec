---
format: aep.planning-md/3
id: epic:population-training
kind: epic
status: draft
title: 'Population training: many small models on one GPU, selected by fitness'
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
revision: 1
---
# Population training on one GPU

## Idea

The operator's idea, 2026-10-22: train many small models on the GPU at once and use them for
self-improvement by selection. Generate N candidate configurations, train them side by side, keep
the best two, derive the next candidates from those, and repeat. That is a genetic algorithm over a
fleet of small models.

- Near term: 4 concurrent trainings of the MNIST benchmark model (784-64-10).
- Target: 100-200 models of the size of the MNIST benchmark model, or smaller, resident on the
  RTX 3090 at once.

## Why it looks feasible (arithmetic, not measured)

- 784-64-10 without bias has 784·64 + 64·10 = 50,816 f32 parameters, about 203 KB.
- 200 copies come to about 41 MB of parameters. With gradients and batch-100 activations it is
  still under 1% of the card's 24 GB.
- Memory is therefore not the limit. The likely limit is per-model dispatch and submission
  overhead: 200 separate processes or plans would each pay the launch cost of a small step.

## Open questions

1. Separate processes, one plan per model, or one plan with a population axis? With the population
   axis, every kernel runs over P models in one dispatch, P weight sets in one buffer, and per-model
   hyperparameters as data. The third option is the one that scales to 200. It probably touches the
   Kernel IR and the execution plan (DESIGN.md § 3.1, § 8), which makes it a design change.
2. What varies between candidates: initial weights, hyperparameters (learning rate, batch size),
   architecture (layer widths), or the `.mspec` source itself? Architecture changes break the
   "one plan, population axis" shape unless the shapes are padded or grouped.
3. Selection: what is the fitness (eval loss, accuracy, accuracy per step)? How is the next
   generation produced (copy and perturb, crossover of weights, mutation of hyperparameters)?
4. Record: each candidate's training run, and each generation's selection, should leave a typed
   record in `spec/` (O6), so a generation can be replayed and compared.
5. Numerics: on the RTX 3090 the tolerance class applies
   (`approval-record:m1-tolerance-numerics-class`). Is selection stable across reruns when two
   candidates differ by less than the tolerance?

## Depends on

- `epic:m1-vulkan-execution`: training on the GPU at all (story `m1-vulkan-train`).
- `story:train-data-dense-tensors`: dense data, `--init` and `--eval` give a candidate its data,
  weights and fitness.

## First step when picked up

Measure before designing. Run 1, 2, 4, 8 and 16 concurrent `modelspec train --backend vulkan` on
the benchmark and record µs/step per model and in total, with the GPU otherwise idle. Do this once
M1 trains on the GPU. The curve shows whether separate processes are enough for the near-term 4 and
where the population axis starts to pay.
