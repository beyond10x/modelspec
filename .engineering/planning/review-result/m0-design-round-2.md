---
format: aep.planning-md/3
id: review-result:m0-design-round-2
kind: review-result
status: active
title: Design critic, M0 decomposition round 2
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

Both round-1 findings are fixed. I found two new seams in the sgd story's inputs.

**Round-1 findings now closed**
- `TrainingProgram` to interpreter: `m0-reverse-mode-autodiff.md:30-35` now says `TrainingProgram` holds `program: TensorProgram` and a caller passes `&training.program`. It also says no overload or accessor is added. `m0-cpu-sgd-training.md:57-60` runs `&training.program` unchanged.
- Effect tokens and state versions: `m0-cpu-sgd-training.md:28-34` defines them as Training IR metadata in `ir/training.rs`. Line 35 puts the legal-effects check in `ir/verify.rs`, extending the autodiff checks. `m0-cpu-interpreter.md:43-45` states they are never interpreter ops. The "if it needs them" ping-pong is gone.

**New findings**
- story:m0-check-types-shapes — its `hir.rs` entry says only that Typed HIR carries "each parameter carrying its canonical path". It does not say HIR holds the `train` block (`steps`, `lr`, `seed`, `input`/`target`, `loss`). Yet `TrainConfig` is "read from the HIR `train` block" in the sgd story, and the sgd body gives `compile_training` no way to reach the HIR. `compile_tensor` returns only a `TensorProgram`. The interface is owned by no body. State in `hir.rs` that HIR carries the train block, and say in the sgd story how `compile_training` obtains it. No new edge is needed: sgd depends on lower, which depends on check — `.engineering/planning/story/m0-check-types-shapes.md:36-37`, `.engineering/planning/story/m0-cpu-sgd-training.md:33-34,39-40`, `.engineering/planning/story/m0-lower-tensor-ir.md:29`
- story:m0-cpu-sgd-training — `data.rs` computes targets as "the model `forward` under teacher parameters, computed by the interpreter". The lowered program has the loss as its single output with `forward` inlined, and `TrainingProgram` records only the loss and the gradients. No body exposes a forward-only program or the prediction `ValueId`, so the sgd story's data half cannot be described without lower's internals. The body should say how the prediction value is obtained. Options: lower records it, or `compile_training` builds a forward-only program — `.engineering/planning/story/m0-cpu-sgd-training.md:49-52`, `.engineering/planning/story/m0-lower-tensor-ir.md:23-26`

I found neither seam in round 1, so both are marked `introduced`. They are not caused by the revision's changes.

**What you read:** 8 stories in full and the round-1 verdict, plus `DESIGN.md` §§ 3.1, 4 and `examples/linear_regression.mspec`. Commands: `aep plan artifact list --kind story`, `show` on all 8 and on the round-1 review-result, `relations`, `graph` and `validate` (valid). I walked 28 non-review edges, 12 `depends_on` among the stories plus 16 `decomposes` and `serves` edges to the epic and vision outside the set. There is no cycle. The set is not a queue: parse → check → lower runs beside tensor-ir-core → interpreter → autodiff, and the two join at sgd. I traced 14 acceptance reads to a producer in the set. All 14 are covered by a `depends_on` edge, 5 only transitively. I found no hidden dependency and no horizontal slice.

**What I could not establish**
- Out of my lane (parallel safety): `Taskfile.yml`, `ir/verify.rs`, `ir/training.rs`, `ir/print.rs`, `compile.rs`, `ir.rs` and `DESIGN.md` are each edited by more than one story. Each pair that shares a file has a `depends_on` edge, except `Taskfile.yml`, `ir.rs` and `DESIGN.md` for lower and interpreter. These two do not touch `Taskfile.yml`, so I leave it there.
- Out of my lane (scope): `m0-cpu-sgd-training.md` says to "record this key, counter and stream derivation" in DESIGN.md § 2.3 and bump the revision line. The interpreter also bumps it, so I judged no revision number is hard-coded now.
- Unease, not a finding: the sgd story still bundles the Training IR compiler half with the training host half. It is last on the longest chain.

```findings
- file: .engineering/planning/story/m0-check-types-shapes.md
  line: 36
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the hir.rs entry does not say Typed HIR carries the train block (steps, lr, seed, input and target, loss) that TrainConfig in story m0-cpu-sgd-training is read from, and compile_tensor returns only a TensorProgram, so the HIR-to-TrainConfig interface is owned by no body"
- file: .engineering/planning/story/m0-cpu-sgd-training.md
  line: 49
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "data.rs computes targets as the model forward under teacher parameters, but the lowered program has the loss as its single output and TrainingProgram records no prediction value, so the body does not say how a forward-only program or the prediction ValueId is obtained"
```
