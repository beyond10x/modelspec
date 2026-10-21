---
format: aep.planning-md/3
id: review-result:m0-design-round-1
kind: review-result
status: active
title: Design critic, M0 decomposition round 1
relations:
- reviews: story:m0-check-types-shapes
- reviews: story:m0-cpu-interpreter
- reviews: story:m0-cpu-sgd-training
- reviews: story:m0-lower-tensor-ir
- reviews: story:m0-parse-subset
- reviews: story:m0-reverse-mode-autodiff
- reviews: story:m0-tensor-ir-core
- reviews: story:m0-workspace-foundation
revision: 1
---
needs-revision

- story:m0-reverse-mode-autodiff — the body never says how the CPU interpreter, which "evaluates a `TensorProgram`", runs a `TrainingProgram` (a wrapped `TensorProgram`, an accessor, or an interpreter overload), yet `gradients.rs` evaluates "each gradient program" on that interpreter and story:m0-cpu-sgd-training later runs the "training program" there. The seam between the autodiff output type and the interpreter input type belongs to neither body, and the existing depends_on edge records order but not the interface. State it in this body, as the type `TrainingProgram` exposes to the interpreter — .engineering/planning/story/m0-reverse-mode-autodiff.md:33,37-39; .engineering/planning/story/m0-cpu-interpreter.md:29
- story:m0-cpu-sgd-training — Training IR is split across three stories, and its half does not say where effect-token and state-version ops are defined or which verifier checks cover them. `TensorOp` lives in tensor.rs (owned by m0-tensor-ir-core) and the Training IR checks in verify.rs belong to m0-reverse-mode-autodiff. This body edits neither file, though `compile_training` ends "then verify". m0-cpu-interpreter defers the same ops to this story "if it needs them", and this story defers back with "if the interpreter needs them", so each side makes the other conditional. The body should name the defining file, a verify.rs extension for effect ordering and no write before the last read, and an unconditional owner for the interpreter's state ops. No new edge is needed, since depends_on edges to both already exist — .engineering/planning/story/m0-cpu-sgd-training.md:30-33,51-52; .engineering/planning/story/m0-cpu-interpreter.md:44; .engineering/planning/story/m0-reverse-mode-autodiff.md:35

**What you read.** 8 stories, the epic, DESIGN.md §§ 2.2–4, 6–9 and AGENTS.md, and `examples/linear_regression.mspec`. Commands: `aep plan artifact list --kind story`, `relations`, `graph`, `validate` (reports valid), and `show` on all 8 stories plus the epic. I walked all 31 edges in the graph, including those to the epic, the vision, the ESS artifact and `epic:m1-vulkan-execution`. There is no cycle, and the edges are acyclic `depends_on`. The set is not a single queue: the parse → check → lower branch runs beside the tensor-ir-core → interpreter → autodiff branch and the two join at m0-cpu-sgd-training. I traced 14 acceptance reads to a producer in the set. All 14 are recorded by a depends_on edge, 4 of them only transitively (for example the foundation files and `diag.rs` read by check and lower). I found no hidden dependency or horizontal slice.

**What I could not establish.**
- Out of my lane (acceptance or scope): m0-cpu-interpreter says it "checks every supplied tensor against its declared type" (line 30), but its acceptance and the gradient story run `f64` tensors through programs typed `DType::F32` only. Nothing says the dtype part of that check is relaxed.
- Out of my lane (scope): m0-cpu-sgd-training hard-codes "DESIGN.md § 2.3 (revision 0.2)" (line 37). m0-cpu-interpreter bumps the revision line first, because sgd depends on it, so the number will be stale.
- Unease, not a finding: m0-cpu-sgd-training bundles the Training IR compiler half (optimizer subgraph, `ir --stage training`) with the training host half (Philox init, synthetic data, runner). It is the last story on the longest chain. Size is not my lane, and I could not tell whether you want the two halves split.
- Whether `TrainingProgram` and `TensorProgram` are the same SSA container is not decidable from the bodies. DESIGN.md § 3.1 and § 3.2 do not settle it either.

```findings
- file: .engineering/planning/story/m0-reverse-mode-autodiff.md
  line: 33
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the body does not say how the interpreter, which evaluates a TensorProgram, runs a TrainingProgram, although gradients.rs evaluates gradient programs on it and the sgd story later runs the training program there, so the autodiff-to-interpreter interface is owned by no body"
- file: .engineering/planning/story/m0-cpu-sgd-training.md
  line: 30
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "Training IR is split across three stories and this body does not say where effect-token and state-version ops are defined or which verify.rs checks cover them, while the interpreter story defers the same ops to this story 'if it needs them' and this story defers back 'if the interpreter needs them'"
```
