---
format: aep.planning-md/3
id: review-result:m0-parallel-safety-round-1
kind: review-result
status: active
title: Parallel-safety critic, M0 decomposition round 1
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

m0-parse-subset — it and story:m0-tensor-ir-core can run at once, and both change `crates/modelspec-core/Cargo.toml` and `Cargo.lock` without saying so (inferred). Parse adds `logos` and tensor-ir-core adds `thiserror`. Neither lists a Cargo manifest under "Lands in", foundation's `[workspace.dependencies]` names no crates, and no `depends_on` edge joins the two. The two remedies are an ordering edge that records the shared manifest as its reason, or splitting the surface so foundation declares `logos`, `thiserror` and `blake3` up front. — .engineering/planning/story/m0-parse-subset.md:27, .engineering/planning/story/m0-tensor-ir-core.md:42, .engineering/planning/story/m0-workspace-foundation.md:28-29

**What I read**
- 8 stories in full, from `aep plan artifact list --kind story` and `aep plan artifact show` on each.
- `aep plan artifact graph`, `aep plan artifact relations` and `aep plan artifact waves`.
- DESIGN.md §7 (lines 384-389) and `examples/linear_regression.mspec`.
- `git status` and a grep of the story bodies for Cargo, `logos`, `thiserror`, `blake3` and `clap`.

**Surface counts**
- Surfaces placed: 8 cited, 0 inferred, 0 unplaceable. The one finding rests on an inferred collision, because the manifest edit is implied by the dependencies the bodies name, not cited.
- `aep plan artifact waves` reports 0 waves and 8 unassessed, because no story has a recorded scope. That is a store fact, not a defect in the bodies.
- Acceptance reads traced to a producer in the set: 15. A `depends_on` edge records 12 directly. The other 3 are recorded only transitively:
  - check-types reads foundation's `diag.rs`.
  - cpu-interpreter reads foundation's runtime stubs.
  - cpu-sgd-training reads the train-block configuration that check-types' `hir.rs` carries.

**Collisions I judged acceptable**

Each shared file below is ordered by an existing edge, and the bodies admit the sharing in most cases:
- `ops.rs`: tensor-ir-core, then autodiff. Check-types only reads it and says so.
- `ir/verify.rs`: tensor-ir-core, then autodiff.
- `ir/training.rs`: autodiff, then cpu-sgd-training.
- `compile.rs` and `ir/print.rs`: lower-tensor-ir, then cpu-sgd-training.
- `cpu/interpreter.rs`: cpu-interpreter, then cpu-sgd-training.
- `cli/ir.rs`: lower-tensor-ir, then cpu-sgd-training.
- DESIGN.md §2.2 and §2.3 with the revision line: cpu-interpreter, then cpu-sgd-training.
- `AGENTS.md` and `Taskfile.yml`: foundation, then check-types.
- lower-tensor-ir and autodiff are the only pair the bodies say may run side by side. Their surfaces are disjoint, and lower-tensor-ir's "Out of scope" names the files it will not touch.

**What I could not establish**
- Whether foundation's `diag.rs` needs `thiserror` or other crates. Foundation lists no dependencies, so I could not tell if the collision above also reaches foundation.
- Out of my lane, for the design critic: `compile_tensor` returns only a `TensorProgram`. cpu-sgd-training reads `steps`, `lr` and `seed` from the `train` block ("SGD configuration from the file"). The bodies do not say which type carries them from `hir.rs` to the runtime.

```findings
- file: .engineering/planning/story/m0-parse-subset.md
  line: 27
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "this story and story:m0-tensor-ir-core (line 42 of its body) can run at once and both change crates/modelspec-core/Cargo.toml and Cargo.lock (adds logos, adds thiserror) without saying so, no depends_on edge orders them and foundation's [workspace.dependencies] names no crates; surface is inferred, not cited; remedies are an ordering edge recording the manifest as its reason, or foundation declaring the dependencies up front"
```
