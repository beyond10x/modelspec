---
format: aep.planning-md/3
id: story:muon-optimizer
kind: story
status: proposed
title: 'Muon: orthogonalized momentum for the matrix parameters, compiled into the training step'
relations:
- decomposes: epic:bert-from-scratch
- depends_on: story:bert-adamw-warmup
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/tests/check_muon.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/
- confidence: cited
  path: crates/modelspec-cli/tests/train_muon.rs
- confidence: cited
  path: crates/modelspec-core/src/compile.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/training.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/ast.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-runtime/src/cpu/interpreter.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/optimizer.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/muon.rs
- confidence: cited
  path: opponents/**
- confidence: cited
  path: spec/domains/training.yaml
revision: 3
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T11:24:37Z", actor: "human:timo", revision: 3}
---
# Muon: orthogonalized momentum for the matrix parameters, compiled into the training step

## Context

The operator asked on 2026-10-25 to investigate "a newer optimizer better than AdamW especially
for transformers, something like muon", and to integrate it if it fits the goal "modelspec
trains simple transformer from scratch with equal performance as python SOTA approach".

Muon (MomentUm Orthogonalized by Newton-Schulz, Jordan et al., 2024) keeps SGD momentum per
matrix parameter and replaces the momentum update by its nearest semi-orthogonal matrix,
computed with five Newton-Schulz iterations (three matmuls each) instead of an SVD. It applies to
the rank-2 hidden weights only; embeddings, the decoder, norms and biases keep AdamW. It holds
the modded-nanoGPT speed records and PyTorch ships it as `torch.optim.Muon`. Moonshot's "Muon is
Scalable for LLM Training" (arXiv 2502.16982, 2025) added decoupled weight decay and an update
scale of `0.2 * sqrt(max(d_out, d_in))` that matches AdamW's update RMS, so AdamW's learning
rate transfers. Measured gains over AdamW: about 1.4x in steps to a loss at models under 0.5B
parameters, 1.1x at 1.2B ("Fantastic Pretraining Optimizers and Where to Find Them", arXiv
2509.02046); Moonshot reports about 2x; Essential AI 1.1 to 1.2x. The models of this epic are
small, where the measured gain is largest.

Why it fits this compiler: every operation Muon needs is already a Tensor IR node with a fixed
reduction order on both backends: `matmul` (the strongest kernel in the repository), `transpose`
(free under a matmul), scalar multiply and add, and a sum of squares, so the Newton-Schulz
iteration compiles into the step's `Optimizer` subgraph the way AdamW did
(`story:bert-adamw-warmup`, DESIGN.md 0.47) with no new kernel kind, and its result is
reproducible bit for bit under ADR-009. Caveats from the literature, which the acceptance
below measures rather than assumes: Newton-Schulz with the fixed coefficients was tuned for
bf16 and gives singular values in about [0.7, 1.3], not 1; the method is reported to work poorly
at small batches; a very rectangular matrix converges worse (the embedding and decoder are
excluded for that reason).

## What this story decides

- Syntax: `optimizer = muon(lr = F, momentum = F, weight_decay = F, ns_steps = N, warmup_steps = N,
  adam_lr = F, beta1 = F, beta2 = F, eps = F)`, every argument required and named, in any order,
  the same parsing rules and codes as `adamw` (E-TYPE-001, E-TYPE-002, E-TYPE-019, E-TYPE-010 for
  `lr` and `adam_lr`, E-TYPE-023 for `weight_decay`, the betas and `eps`, E-TYPE-024 for
  `warmup_steps`); a new E-TYPE-025 refuses `momentum` outside [0, 1) and `ns_steps` outside
  [1, 10]. E-SYNTAX-013 names `muon` among the accepted optimizers.
- The split: a parameter takes the Muon update when its rank is 2 and no `embedding` reads it (a
  tied decoder is read by `embedding` through `transpose`, so it is excluded with its table);
  every other parameter with a gradient takes the AdamW update of `story:bert-adamw-warmup`
  with `adam_lr`, `beta1`, `beta2`, `eps` and the same `weight_decay`. `ir --stage training`
  prints which update each parameter takes.
- The Muon update at step `t` (from 1), per matrix `p` of shape `[d_out, d_in]` with gradient
  `g`: `b = momentum * b + g`; `u = g + momentum * b` (Nesterov); `x = u / (frobenius(u) + 1e-7)`,
  transposed when `d_out > d_in` so the Gram matrix is the smaller one; `ns_steps` times
  `a = x xᵀ`, `x = 3.4445 x + (-4.7750 a + 2.0315 a a) x` with every matmul in the repository's
  order and the scalar operations each one rounded `f32` operation in the written order; `x`
  transposed back; `p = p - lr_t * (0.2 * sqrt(max(d_out, d_in)) * x + weight_decay * p)`, the
  scale a host constant computed in `f64` and rounded once. `lr_t` is the warm-up schedule of
  D50 applied to `lr` for the Muon parameters and to `adam_lr` for the AdamW ones. The momentum
  buffer `b` is optimizer state versioned with the parameter as the AdamW moments are
  (`optimizer.b@<path>`), 0 before the first step.
- No checkpoint and no `--resume` for a Muon run (the AdamW rule, D50), no Vulkan kernel yet
  (E-TARGET-011 at `train --backend vulkan`).
- The PyTorch opponent of the comparison stories gains a `--optimizer muon` mode built on
  `torch.optim.Muon` with the same split and hyperparameters, so "equal performance as the
  python approach" compares the same algorithm on both sides.

## Domain relations

`spec/domains/training.yaml`: `OptimizerConfig.kind` gains `muon`; a `MuonConfig` holds
`momentum`, `ns_steps` and the nested AdamW settings for the excluded parameters; the
`LearningRateSchedule` of AdamW applies unchanged. Schemas regenerated; no observatory change.

## Acceptance

1. One step of `muon(...)` on a 2-parameter model (a `[4, 3]` dense weight and a `[3]` bias,
   `ns_steps = 5`) is bit-equal to an independent `f32` reference written in the test in the
   written order; the bias took the AdamW update, the weight the Muon update.
2. For a random `[64, 32]` gradient, every singular value of the orthogonalized update (computed
   in `f64` in the test from the `f32` result) is within [0.7, 1.3] at `ns_steps = 5`, and the
   update's RMS is within 10 % of `0.2` after the scale.
3. On `crates/modelspec-cli/tests/fixtures/mlm_bag.mspec`, `ir --stage training` lists the token
   table under AdamW and the dense weight under Muon; a model with the decoder tied through
   `transpose(table)` lists the table under AdamW.
4. Each refusal of the syntax at its boundary, with the code and the span at the argument:
   `momentum = 1.0`, `ns_steps = 0`, `ns_steps = 11`, a missing argument, a repeated one.
5. The hidden-32 twin of `story:bert-tiny-cpu-model`, or `mlm_bag.mspec` on the packed probe
   corpus of `story:bert-heldout-evaluation` if the twin has not landed, trains 300 steps on the
   CPU with `--eval-every 50` under `muon` and under `adamw` with the Moonshot-matched rates; the
   held-out loss at step 300 under Muon is at most the AdamW figure, both quoted in the
   report and in DESIGN.md § 2.1 beside the rule.
6. `train --backend vulkan` on a Muon file is E-TARGET-011, exit 2, before anything runs;
   `--checkpoint` and `--resume` exit 2 on one line.
7. `opponents/bert_tiny/` (or the opponent script of the comparison story, if it exists by then)
   runs `--optimizer muon` on the pinned torch and prints its held-out loss at the same steps.

## Out of scope

A Vulkan kernel for the update (the attention/loss/AdamW kernel story takes it with the other
optimizer kernels); checkpointing optimizer state; parameter groups in the language; adaptive
Newton-Schulz coefficients (Polar Express) and the other 2025 to 2026 variants.
