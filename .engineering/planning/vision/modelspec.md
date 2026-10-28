---
format: aep.planning-md/3
id: vision:modelspec
kind: vision
status: approved
title: 'ModelSpec: a declarative differentiable-program compiler with an owned Vulkan runtime'
revision: 6
transitions:
- {from: "draft", to: "in_review", at: "2026-10-21T08:41:05Z", actor: "human:timo", revision: 2}
- {from: "in_review", to: "approved", at: "2026-10-21T08:41:05Z", actor: "human:timo", revision: 3}
---
# ModelSpec: specify what to compute, compile how to run it

## North star

**Train a model much faster than a general-purpose framework can, because the compiler builds a
runtime for exactly one target.** Stated by the operator on 2026-10-28; it is what ModelSpec is
for, and every other section of this vision serves it.

1. **The compiler knows everything before it runs.** The whole training step (model, loss,
   gradients, optimizer, data bindings, execution policy) is one declared, typed program. Nothing
   is decided at run time, so the compiler can compile that one program into a runtime built for
   one target: one use case, one operating system, one GPU chipset. A general-purpose framework
   must stay general while it runs; a ModelSpec runtime does not, and that is where the speed
   comes from.
2. **Aggressive optimization is safe to try.** ModelSpec is specified, and every kernel, fusion and
   pass is checked against the CPU reference interpreter on the same verified IR. A change to the
   compiler, by a person or by an agent, that alters a result is refused; a change that only makes
   it faster is kept. That is what lets the project optimize hard and quickly, and it is the
   practical form of the governed self-improvement direction below.
3. **Where it stands, and the worked example.** One target exists today: Vulkan compute and SPIR-V
   on one NVIDIA RTX 3090 on Linux. ModelSpec is slower than PyTorch there today: the BERT-tiny
   encoder block at batch 8 takes 93,868.4 µs per step against 3,937.2 µs on PyTorch CUDA eager and
   1,971.9 µs on `torch.compile` (release 0.21.0, the Benchmarks page). The ladder from that
   baseline, every step planned:
   - **Baseline**: BERT-tiny against PyTorch on the same RTX 3090, eager and `torch.compile`; the
     block figures are released, the full-model figures land with release 0.22.0.
   - **A**: the transformer block within 3× of PyTorch eager
     (`initiative:transformer-within-3x-of-pytorch`, `epic:transformer-block-within-3x`).
   - **E**: full BERT-tiny at parity with PyTorch eager.
   - **F**: faster than `torch.compile` per step and to the held-out target.

The speed is the goal, not shipped. Every public page that states it labels it planned and gives
the released figures beside it. Target-specific runtimes for other operating systems and chipsets
are planned; "a specific chipset" means a runtime specialised for the device profile, through
Vulkan today, never SASS or CUDA (DESIGN.md § 0).

## Objective

A versioned, declarative language and Rust compiler for differentiable tensor programs whose whole
training step (forward, loss, backward, optimizer) is compiled into a runtime specialised for one
target and run through an independently owned Rust runtime, first on an NVIDIA RTX 3090 through
Vulkan compute and SPIR-V, with no Python, PyTorch or CUDA API/toolkit dependency (DESIGN.md § 0).
Correctness comes first: the specialised runtime is held to the CPU reference, and the speed of
the north star is earned on top of that, one measured step at a time.

## The problem

- A training run today is a Python program over a framework, a CUDA toolkit and a driver. What
  actually ran (shapes, gradient graph, optimizer state, random streams) is spread over runtime
  objects nobody can read back as one artifact.
- The framework cannot see the whole step before it runs, so it stays general at run time and
  pays for that generality on every step.
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

- Engineers who want a model trained faster on the hardware they have, from a training program
  they can read, diff, review and reproduce as one file.
- Tools and agents that generate or change models, or the compiler itself, and need every proposal
  checked by a compiler and a reference before it costs compute.
- Teams that want to own their stack from source to SPIR-V without a Python or CUDA toolkit
  dependency.

## Use cases

1. **Faster training on one target (planned).** A `.mspec` program compiled into a runtime
   specialised for one use case, one operating system and one GPU chipset trains faster than a
   general-purpose framework on the same device; BERT-tiny on the RTX 3090 is the first measure.
2. **Reviewable model changes.** A change to an architecture, loss or optimizer is a diff of a
   `.mspec` file; `modelspec check` refuses a shape or type error with a coded diagnostic before
   any training starts.
3. **Reproducible experiments.** Parameter initial values come from Philox4x32-10 keyed by the
   seed and a `blake3` hash of each parameter's path, not from compiler order (DESIGN.md § 2.3);
   checkpoints carry a `blake3` checksum (§ 6.2). The same source and seed give the same run.
4. **Differential testing of a GPU backend.** Every kernel, fusion and pass is checked against the
   CPU interpreter on the same IR (DESIGN.md § 0 invariants 1 and 7).
5. **Small-model training without Python.** Train MLPs and, from M3, small GPT-like and BERT-like
   models on a consumer GPU through Vulkan.

## Direction: governed recursive self-improvement

These are directions the design makes possible, not milestones. Item 2 is the north star's
practical form and is already how speed work is done (one measured unit at a time, each held to
the CPU reference); the others are not planned in M0–M4.

1. **Agents propose models as spec diffs.** An agent edits a `.mspec` file; the checker rejects an
   invalid proposal in milliseconds, and only checked proposals reach training. The compiler is the
   first gate of the loop.
2. **A compiler that improves its own kernels.** Agents propose passes, fusions and kernel variants;
   each is accepted only if differential tests against the CPU oracle stay green and a benchmark
   improves. Speed can change; semantics cannot change unseen. This is what makes the north star
   reachable quickly: aggressive optimizations are safe to try because a wrong one is refused.
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

- North star: step time and time to a held-out loss of BERT-tiny on the RTX 3090 against PyTorch
  CUDA eager and `torch.compile`, measured the same day on the same device and driver and published
  on the Benchmarks page; the ladder's steps A, E and F above are its checkpoints.
- M0: linear regression trains correctly on the CPU reference interpreter, and a two-layer relu
  graph passes finite-difference gradient checks (DESIGN.md § 1, § 11 iteration 1). Module-based
  MLPs such as `examples/tiny_mlp.mspec` arrive with § 11 iteration 3.
- M1: the first end-to-end f32 training run on the RTX 3090 through Vulkan meets the nine
  acceptance criteria of DESIGN.md § 9.
- M2–M4: memory planning and fusion; f16, AdamW and attention for a small GPT-like model; the
  cooperative-matrix path and autotuning (README.md milestones).

## Constraints

The seven non-negotiable invariants of DESIGN.md § 0. Correctness and coherent architecture come
before speed: no optimization is kept that changes a result beyond the stated tolerance of the CPU
reference. Speed is the north star, pursued one measured step at a time; near-optimal performance
on a wide range of networks is not a v1 guarantee.

## Non-goals

- Generating NVIDIA SASS or replacing the NVIDIA kernel-mode driver (DESIGN.md § 0). A runtime
  "for one chipset" is specialised for the device profile the driver reports, through Vulkan and
  SPIR-V.
- The CUDA API or toolkit as a device path (DESIGN.md § 0; the optional PTX backend of § 12 is an
  open question and would not meet it).
- A PyTorch-compatible API, Python bindings or a general-purpose programming language.
- A general-purpose runtime that adapts at run time; the speed comes from not being one.
- Autonomous self-modification without review: every loop above runs inside the governance in
  direction item 5.
