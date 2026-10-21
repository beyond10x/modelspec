---
format: aep.planning-md/3
id: review-result:m0-scope-round-2
kind: review-result
status: active
title: Scope critic, M0 decomposition round 2
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
approve

**What I read**
- 10 artifacts, each in full: `epic:m0-semantic-core`, the 8 `story:m0-*` stories and `vision:modelspec`.
- `review-result:m0-scope-round-1`, and DESIGN.md §§ 2, 3, 3.2, 3.3 and 9 by grep.
- Commands: `aep plan artifact list --kind story`, `aep plan artifact show` on each, and `aep plan artifact graph`. I did not re-run the validator.
- Promises: I extracted 16 from the epic. That is the outcome chain (parse, check, lower, differentiate, train on CPU), the 5 in-scope bullets split into their sub-clauses, and the 3 acceptance bullets. I traced all 16 to a story.

**Round-1 finding**
- The effect check is now claimed once. `story:m0-cpu-sgd-training` revision 5 names it in its Context ("the effect part of the epic's 'type/shape/effect checks'"). Its Lands in puts the legal-effects check of DESIGN.md § 3.2 in `ir/verify.rs`. Its tests include `crates/modelspec-core/tests/training_verify.rs`, where a write ordered before the last read of the old version is refused.
- `story:m0-check-types-shapes` disclaims it in its Context and keeps no claim to it, so the two stories do not both claim the effect check.
- `story:m0-reverse-mode-autodiff` owns the other Training IR checks (gradient types, gradient dependence on the loss). These are different checks, so nothing is claimed twice.

**Coverage and reach**
- Canonicalization and constant folding are not an epic promise. The epic's in-scope line lists lexer, parser, resolution, checks, Typed HIR, Tensor IR and verifier. The deferral is stated in `story:m0-lower-tensor-ir` (Out of scope) and `story:m0-tensor-ir-core` (Out of scope), so it is a named omission. DCE is kept and traces to the "Tensor IR" promise and DESIGN.md § 3.
- Synthetic teacher data with no `--data` flag traces to the epic's "approaches the known solution" and DESIGN.md line 451 ("synthetic linear regression from fixed seeds"). The cut is named in `story:m0-cpu-sgd-training` (Out of scope).
- `tiny_mlp.mspec`, modules and generics are excluded in four stories, and `vision:modelspec` revision 4 puts them at § 11 iteration 3. The epic does not promise them.
- The two-layer relu graph in `story:m0-reverse-mode-autodiff` is grounded in the vision's M0 measure and DESIGN.md § 9 gate 3. It is not reach beyond the parent.
- The extras are each grounded in DESIGN.md or AGENTS.md:
  - `ir --stage training`, which is DESIGN.md § 8;
  - the `check-examples` and `core-no-vulkan` Taskfile tasks;
  - the DESIGN.md § 2.2 and § 2.3 sentences with a revision bump.
- No outcome is claimed twice. The shared files (`ops.rs`, `ir/verify.rs`, `ir/training.rs`, `compile.rs`, `ir/print.rs`, `ir.rs`) get different outcomes, and each pair is ordered by `depends_on` or an explicit out-of-scope line.

**What I could not establish**
- I was not given the drafter's report. I judged against the stories' own Out of scope sections and the decisions you listed.
- Out of my lane, and not setting the verdict:
  - the `story:m0-cpu-sgd-training` margin claim (parameter error near 2e-4 after 500 steps, eigenvalue 0.17) is an acceptance-checkability question;
  - `story:m0-lower-tensor-ir` and `story:m0-reverse-mode-autodiff` may run beside each other. `story:m0-lower-tensor-ir` says `compile.rs` and `ir.rs` are its own, and `story:m0-cpu-sgd-training` extends both later, so sequencing is a parallel-safety question.

```findings
[]
```
