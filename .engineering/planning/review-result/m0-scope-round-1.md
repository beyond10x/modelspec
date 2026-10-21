---
format: aep.planning-md/3
id: review-result:m0-scope-round-1
kind: review-result
status: active
title: Scope critic, M0 decomposition round 1
relations:
- reviews: epic:m0-semantic-core
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

epic:m0-semantic-core — "type/shape/effect checks" and "verifier" are promised, and no story's Lands in or Acceptance claims an effect check. `story:m0-check-types-shapes` only cites "effect checks" in its Context (line 20), and the training-state effect-ordering legality check (DESIGN.md § 3.2 verifier, "legal effects") has no verifier owner. Either `story:m0-check-types-shapes` or `story:m0-cpu-sgd-training` should name the effect check in its Lands in, or say it is deferred and why. — `.engineering/planning/epic/m0-semantic-core.md:24`; `.engineering/planning/story/m0-check-types-shapes.md:20`; `.engineering/planning/story/m0-cpu-sgd-training.md:30`

**What I read**
- 10 artifacts: the epic and all 8 `story:m0-*` stories, each in full.
- Context read: `epic:m1-vulkan-execution` (via the graph), DESIGN.md §§ 0–4, 6–9, 11–12, AGENTS.md, `examples/linear_regression.mspec`, and `.engineering/drafts/` for any drafter note.
- Commands: `aep plan artifact list --kind story`, `aep plan artifact show` on the epic and every story, `aep plan artifact graph`, `aep plan artifact validate` (reports valid), `grep -i effect` over the stories and drafts.
- Promises: I extracted 15 from the epic (outcome, 5 in-scope bullets, 3 acceptance bullets, and sub-clauses) and traced 14 to a story. The one untraced promise is the effect-check clause above.
- Reach beyond the parent: none found. The extras are each grounded in DESIGN.md or AGENTS.md:
  - the two-layer relu graph (§ 9 gate 3);
  - `ir --stage training` (§ 8);
  - the DESIGN.md § 2.2 and § 2.3 sentences the stories add, with a revision bump (AGENTS.md rule);
  - the `check-examples` Taskfile task (AGENTS.md rule on examples).
- Double claims: none. The shared files (`ir/training.rs`, `compile.rs`, `ir/print.rs`) are each extended with a different outcome.
- Narrowed promises: none besides the effect clause. The drafter names the other omissions (canonicalization, modules and generics, checkpoints, a data format).

**What I could not establish**
- The drafter's report was not given to me. I looked in `.engineering/drafts/` and found only the story drafts, so I cannot tell whether the effect checks were deliberately left out.
- Out of my lane, and not setting the verdict:
  - the `m0-cpu-sgd-training` margin claims, such as a parameter error near 2e-4 after 500 steps (acceptance);
  - the `m0-reverse-mode-autodiff` and `m0-lower-tensor-ir` notes about running beside each other and who owns `ops.rs`, `verify.rs` and `training.rs` (parallel-safety);
  - whether `m0-cpu-interpreter` and `m0-reverse-mode-autodiff` depending on each other leaves no cycle (design).

```findings
- file: .engineering/planning/epic/m0-semantic-core.md
  line: 24
  category: scope
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "the epic promises \"type/shape/effect checks\" and a verifier, and no story's Lands in or Acceptance claims an effect check: story:m0-check-types-shapes only cites \"effect checks\" in its Context (line 20) and no story verifies training-state effect-token legality (DESIGN.md § 3.2 \"legal effects\"); story:m0-check-types-shapes or story:m0-cpu-sgd-training should claim it or record it as deferred"
```
