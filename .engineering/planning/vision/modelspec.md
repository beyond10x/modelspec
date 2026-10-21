---
format: aep.planning-md/3
id: vision:modelspec
kind: vision
status: approved
title: 'ModelSpec: a declarative differentiable-program compiler with an owned Vulkan runtime'
revision: 5
transitions:
- {from: "draft", to: "in_review", at: "2026-10-21T08:41:05Z", actor: "human:timo", revision: 2}
- {from: "in_review", to: "approved", at: "2026-10-21T08:41:05Z", actor: "human:timo", revision: 3}
---
# ModelSpec: specify what to compute, compile how to run it

## Objective

A versioned, declarative language and Rust compiler for differentiable tensor programs. A whole
training step (forward, loss, backward, optimizer) is compiled and run through an independently
owned Rust runtime, first on an NVIDIA RTX 3090 through Vulkan compute and SPIR-V, with no Python,
PyTorch or CUDA API/toolkit dependency (DESIGN.md § 0).

## The problem

- A training run today is a Python program over a framework, a CUDA toolkit and a driver. What
  actually ran (shapes, gradient graph, optimizer state, random streams) is spread over runtime
  objects nobody can read back as one artifact.
- Shape and dtype errors surface at run time, often after GPU time has been spent.
- Reproducing a run means reproducing an environment, not re-reading a file.
- A program that writes or edits models (a person, a script or an agent) has no small, typed target
  it can check before it runs.

## What ModelSpec is

- A `.mspec` file declares the model, the loss, the optimizer, the data bindings and the execution
  policy. It is the whole training step, and nothing else is needed to run it.
- The compiler checks names, types and shapes before anything runs, lowers to one verified tensor
  IR, derives the backward pass, and compiles forward, loss, backward and optimizer into one
  `train_step` program.
- One semantic core: the CPU reference interpreter and the GPU backend consume the same verified
  IR, so every GPU result has a reference to be compared against.
- Randomness, parameters and optimizer updates have explicit identities and effects; a seed and the
  source determine the run.

## Who it is for

- Engineers who want a training program they can read, diff, review and reproduce as one file.
- Tools and agents that generate or change models and need every proposal checked by a compiler
  before it costs compute.
- Teams that want to own their stack from source to SPIR-V without a Python or CUDA toolkit
  dependency.

## Use cases

1. **Reviewable model changes.** A change to an architecture, loss or optimizer is a diff of a
   `.mspec` file; `modelspec check` refuses a shape or type error with a coded diagnostic before
   any training starts.
2. **Reproducible experiments.** Parameter initial values come from Philox4x32-10 keyed by the
   seed and a `blake3` hash of each parameter's path, not from compiler order (DESIGN.md § 2.3);
   checkpoints carry a `blake3` checksum (§ 6.2). The same source and seed give the same run.
3. **Differential testing of a GPU backend.** Every kernel, fusion and pass is checked against the
   CPU interpreter on the same IR (DESIGN.md § 0 invariants 1 and 7).
4. **Small-model training without Python.** Train MLPs and, from M3, small GPT-like models on a
   consumer GPU through Vulkan.

## Direction: governed recursive self-improvement

These are directions the design makes possible, not milestones. None is planned in M0–M4.

1. **Agents propose models as spec diffs.** An agent edits a `.mspec` file; the checker rejects an
   invalid proposal in milliseconds, and only checked proposals reach training. The compiler is the
   first gate of the loop.
2. **A compiler that improves its own kernels.** Agents propose passes, fusions and kernel variants;
   each is accepted only if differential tests against the CPU oracle stay green and a benchmark
   improves. Speed can change; semantics cannot change unseen.
3. **Self-tuning training recipes.** Optimizer, schedule and initialization are part of the spec, so
   a search loop changes them as data and records every run, its source and its seed as evidence.
4. **Models that write ModelSpec.** A model trained by ModelSpec generates and repairs `.mspec`
   programs; the typed, small language makes its output checkable, which closes the loop from
   model to trainer to model.
5. **The loop is governed.** Every proposed improvement goes through the same process ModelSpec is
   built with: a planned story, an implementor, an adversarial review, a gate and recorded
   evidence (AEP, ESS). Changing an invariant, starting a device run or widening what the loop may
   change stays a human decision.

## Measures

- M0: linear regression trains correctly on the CPU reference interpreter, and a two-layer relu
  graph passes finite-difference gradient checks (DESIGN.md § 1, § 11 iteration 1). Module-based
  MLPs such as `examples/tiny_mlp.mspec` arrive with § 11 iteration 3.
- M1: the first end-to-end f32 training run on the RTX 3090 through Vulkan meets the nine
  acceptance criteria of DESIGN.md § 9.
- M2–M4: memory planning and fusion; f16, AdamW and attention for a small GPT-like model; the
  cooperative-matrix path and autotuning (README.md milestones).

## Constraints

The seven non-negotiable invariants of DESIGN.md § 0. Correctness and coherent architecture come
before speed; near-optimal performance is a long-term goal, not a v1 guarantee.

## Non-goals

- Generating NVIDIA SASS or replacing the NVIDIA kernel-mode driver (DESIGN.md § 0).
- A PyTorch-compatible API, Python bindings or a general-purpose programming language.
- Autonomous self-modification without review: every loop above runs inside the governance in
  direction item 5.
