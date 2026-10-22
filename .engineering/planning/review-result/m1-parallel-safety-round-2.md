---
format: aep.planning-md/3
id: review-result:m1-parallel-safety-round-2
kind: review-result
status: active
title: Plan critic, parallel-safety lens, round 2, decomposition of epic:m1-vulkan-execution
relations:
- reviews: epic:m1-vulkan-execution
- reviews: story:m1-checkpoint-cpu
- reviews: story:m1-compile-artifact
- reviews: story:m1-inspect-device
- reviews: story:m1-kernel-plan
- reviews: story:m1-mlp-example
- reviews: story:m1-vulkan-checkpoint-infer
- reviews: story:m1-vulkan-elementwise
- reviews: story:m1-vulkan-profile
- reviews: story:m1-vulkan-reduce-matmul
- reviews: story:m1-vulkan-train
revision: 1
---
needs-revision

- story:m1-checkpoint-cpu — it and unordered `story:m1-inspect-device` both claim root `Cargo.toml`, `Cargo.lock`, `crates/modelspec-runtime/Cargo.toml`, `crates/modelspec-cli/src/lib.rs` and `crates/modelspec-docs/src/diagnostics.rs` (cited). Both add `serde` and `serde_json` to `[workspace.dependencies]`, so two branches can add the same key. Neither body names the other, and no edge or sentence orders them. Add an ordering edge that records those files as its reason, or split the surface so only one story adds `serde` and `serde_json`. — .engineering/planning/story/m1-checkpoint-cpu.md:108-111, .engineering/planning/story/m1-inspect-device.md:127-131
- story:m1-checkpoint-cpu — it and unordered `story:m1-vulkan-elementwise` both claim root `Cargo.toml`, `Cargo.lock` and `crates/modelspec-runtime/Cargo.toml` (cited). Neither says so, and no edge runs between them. `story:m1-vulkan-reduce-matmul` is also unordered with this story, but they share no file. Add an edge naming the three files, or split. — .engineering/planning/story/m1-checkpoint-cpu.md:108-111, .engineering/planning/story/m1-vulkan-elementwise.md:100-103
- story:m1-inspect-device — it and unordered `story:m1-kernel-plan` both claim `crates/modelspec-core/Cargo.toml`, `crates/modelspec-core/src/lib.rs` and `Cargo.lock` (cited), and both regenerate `website/docs/reference/cli.md`. The scope names the in-flight stories only, so this in-set pair is unrecorded. Add an edge naming those files, or split. — .engineering/planning/story/m1-inspect-device.md:127-131, .engineering/planning/story/m1-kernel-plan.md:129-130
- story:m1-inspect-device — it and unordered `story:m1-mlp-example` both edit `AGENTS.md` (cited: State on one side, the `task check-examples` line on the other). Neither body says the sections are disjoint, and no edge orders them. Add an edge, or a sentence that records the two sections. — .engineering/planning/story/m1-inspect-device.md:148, .engineering/planning/story/m1-mlp-example.md:73
- story:m1-compile-artifact — it and unordered `story:m1-vulkan-train` both claim `crates/modelspec-docs/src/diagnostics.rs` and the generated `diagnostics.md` and `cli.md` (cited). The scope names `story:m1-checkpoint-cpu` as the only other editor of `diagnostics.rs`. Both stories depend on `story:m1-vulkan-reduce-matmul` and `story:m1-checkpoint-cpu`, and nothing orders them against each other. Add an edge naming `diagnostics.rs`, or split. — .engineering/planning/story/m1-compile-artifact.md:96-98, .engineering/planning/story/m1-vulkan-train.md:133

Round-1 findings of mine:

| Round-1 finding | Status | Evidence |
|---|---|---|
| m1-checkpoint-cpu (`runner.rs`, `train.rs`, `lib.rs`, docs files) | resolved | edges train → checkpoint-cpu and compile-artifact → checkpoint-cpu; scope sentences at m1-checkpoint-cpu.md:116 and m1-compile-artifact.md:96 |
| m1-vulkan-train vs observatory-live-event-stream | resolved | edge plus sentence, m1-vulkan-train.md:~129 |
| m1-vulkan-profile vs m1-vulkan-checkpoint-infer | resolved | edge checkpoint-infer → profile; sentence in both scopes |
| m1-inspect-device scope and Cargo | partly resolved | paths claimed and in-flight pairs stated; the in-set pairs above are still unnamed |
| m1-vulkan-elementwise (`Taskfile.yml`, `AGENTS.md`, `check.yml`, `Cargo.toml`) | resolved | edges to mlp-example and inspect-device, which depends on live-event-stream; sentence at m1-vulkan-elementwise.md:~108 |
| m1-vulkan-reduce-matmul (DESIGN.md) | resolved | R2 decision for the revision line; sentence for `story:observatory-input-adjoint` |
| m1-kernel-plan (`website/docs/reference` granularity) | resolved | files named; live-event-stream overlap stated |

What I read: 10 drafts, `epic:m1-vulkan-execution`, the three in-flight stories (head only), my round-1 record and `revision-1-decisions.md`. I ran `aep plan artifact list --kind story`, `show` on each, and `waves --kind story`, filtered to the m1 and in-flight ids. I took the unordered pairs from the transitive `depends_on` closure.

Placement: 10 of 10 items have a cited Scope section, 0 placed by inference alone (only new-file paths are inferred), 0 unplaceable. I traced 18 acceptance reads to a producer in the set, and all 18 are ordered: 13 by a direct `depends_on` edge, 5 through the chain. No reader-beside-producer defect. All 10 m1 stories depend on `story:observatory-live-event-stream`, directly or through the chain. In-flight `story:cpu-matmul-row-order` is stated in three scopes and `story:observatory-input-adjoint` in `story:m1-vulkan-reduce-matmul`.

What I could not establish:
- Whether the `crates/modelspec-core/src/lib.rs` and `Cargo.toml` line edits would textually conflict. I did not simulate a merge, so the findings rest on the claims, not on a test merge.
- Out of my lane, design critic: the unordered pair `story:m1-checkpoint-cpu` and `story:m1-vulkan-reduce-matmul` shares no file, so it is fine.
- Out of my lane: `story:m1-checkpoint-cpu` and `story:m1-inspect-device` both edit § 8 flag and code prose in DESIGN.md beyond the verb table. R2 does not reopen this, and I record it only.

```findings
- file: .engineering/planning/story/m1-checkpoint-cpu.md
  line: 108
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "this story and unordered story:m1-inspect-device both claim root Cargo.toml, Cargo.lock, crates/modelspec-runtime/Cargo.toml, crates/modelspec-cli/src/lib.rs and crates/modelspec-docs/src/diagnostics.rs (cited, both bodies) and both add serde and serde_json to [workspace.dependencies]; neither body names the other and no edge orders them (add an ordering edge recording these files, or split so one story adds the dependencies)"
- file: .engineering/planning/story/m1-checkpoint-cpu.md
  line: 108
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "this story and unordered story:m1-vulkan-elementwise both claim root Cargo.toml, Cargo.lock and crates/modelspec-runtime/Cargo.toml (cited, both bodies); neither body names the other and no edge orders them (add an ordering edge naming the files, or split)"
- file: .engineering/planning/story/m1-inspect-device.md
  line: 130
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "this story and unordered story:m1-kernel-plan both claim crates/modelspec-core/Cargo.toml, crates/modelspec-core/src/lib.rs and Cargo.lock (cited) and both regenerate website/docs/reference/cli.md; the scope names only the in-flight stories and no edge orders the pair (add an ordering edge naming the files, or split)"
- file: .engineering/planning/story/m1-inspect-device.md
  line: 148
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "this story edits AGENTS.md State and unordered story:m1-mlp-example edits the task check-examples line of AGENTS.md (cited); neither body says the sections are disjoint and no edge orders them (add an ordering edge or a sentence recording the two sections)"
- file: .engineering/planning/story/m1-compile-artifact.md
  line: 96
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "this story and unordered story:m1-vulkan-train both claim crates/modelspec-docs/src/diagnostics.rs and the generated diagnostics.md and cli.md (cited); the scope names only story:m1-checkpoint-cpu as the other editor and no edge orders the pair (add an ordering edge naming diagnostics.rs, or split)"
```
