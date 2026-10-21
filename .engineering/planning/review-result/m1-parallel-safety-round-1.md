---
format: aep.planning-md/3
id: review-result:m1-parallel-safety-round-1
kind: review-result
status: active
title: Plan critic, parallel-safety lens, round 1, decomposition of epic:m1-vulkan-execution
relations:
- reviews: epic:m1-vulkan-execution
revision: 1
---
needs-revision

Findings, one line each:

- story:m1-checkpoint-cpu — its `train/runner.rs` and `cli/src/train.rs` edits (resume, `--steps`, checkpoint flags) land on the files `story:m1-vulkan-train` (Vulkan branch), `story:m1-vulkan-profile` (`--profile`) and in-flight `story:observatory-live-event-stream` (per-step observer in `run`) also edit; no `depends_on` edge, and no body names the others, so add an ordering edge recording `runner.rs`/`train.rs` as the reason or split the surface. The same story also shares `cli/src/lib.rs`, `docs/src/cli.rs` and `docs/src/diagnostics.rs` with unordered `story:m1-compile-artifact`. All cited. — .engineering/drafts/m1-checkpoint-cpu.md:55
- story:m1-vulkan-train — it edits `runtime/src/train/runner.rs` and `cli/src/train.rs`, as does in-flight `story:observatory-live-event-stream` (cited in both bodies, `aep plan artifact show story:observatory-live-event-stream`), and no edge or sentence records it. Add an ordering edge naming those two files, or split the surface. — .engineering/drafts/m1-vulkan-train.md:43
- story:m1-vulkan-profile — it and `story:m1-vulkan-checkpoint-infer` both depend only on `story:m1-vulkan-train`, and both edit `cli/src/train.rs` (cited) and the Vulkan runner's record/submit loop under `runtime/src/vulkan/` (inferred directory). Neither body mentions the other, and waves only separates them (19 and 20) by scope. Add an edge, or split the runner module. — .engineering/drafts/m1-vulkan-profile.md:39
- story:m1-inspect-device — its scope omits `crates/modelspec-core/Cargo.toml` (inferred: the body puts `DeviceProfile` and its JSON form in core and adds `serde` at the workspace root), which `story:m1-kernel-plan` and `story:m1-compile-artifact` claim (cited), so `waves` prints no collision. No m1 story claims `Cargo.lock`, though five add dependencies (`ash`, `serde`, `serde_json`, `rspirv`, `blake3` into core). In-flight `story:observatory-live-event-stream` claims `Cargo.toml`, `crates/modelspec-cli/Cargo.toml` and `Cargo.lock` (cited, inferred), and `story:cpu-matmul-row-order` (active) claims `crates/modelspec-runtime/Cargo.toml`, which `story:m1-inspect-device`, `story:m1-vulkan-elementwise` and `story:m1-checkpoint-cpu` also edit. Add the missing paths and an edge or split for each pair. — .engineering/drafts/m1-inspect-device.md:45
- story:m1-vulkan-elementwise — `Taskfile.yml` and `AGENTS.md` are claimed (cited) by unordered `story:m1-mlp-example` and in-flight `story:observatory-live-event-stream`. `.github/workflows/check.yml` (cited here, inferred there) and root `Cargo.toml` are shared with the same in-flight story. The body names none of them. Add an edge or split. — .engineering/drafts/m1-vulkan-elementwise.md:52
- story:m1-vulkan-reduce-matmul — it edits DESIGN.md § 3.3 (the "reassociate" sentence at `DESIGN.md:224`), and in-flight `story:observatory-input-adjoint` edits § 3.3 pass 4 and § 4 (cited). Both also bump the single revision line (`DESIGN.md:4`), as do seven other m1 drafts (`story:m1-inspect-device`, `story:m1-kernel-plan`, `story:m1-vulkan-elementwise`, `story:m1-compile-artifact`, `story:m1-vulkan-train`, `story:m1-checkpoint-cpu`, `story:m1-vulkan-profile`). Only the in-chain pairs are ordered, and the adjoint has no edge to any m1 story. Name an ordering edge on DESIGN.md or split the revision bump. — .engineering/drafts/m1-vulkan-reduce-matmul.md:40
- story:m1-kernel-plan — six m1 stories claim the directory entry `website/docs/reference` (`story:m1-inspect-device`, `story:m1-kernel-plan`, `story:m1-compile-artifact`, `story:m1-checkpoint-cpu`, `story:m1-vulkan-train`, `story:m1-vulkan-profile`). `story:observatory-live-event-stream` claims the files `website/docs/reference/cli.md` and `crates.md` and the directory `crates/modelspec-cli/tests/`. `waves` compares strings and prints no collision, yet both regenerate `cli.md` from clap. State the overlap, or claim one granularity throughout. — .engineering/drafts/m1-kernel-plan.md:68

What I read: 10 drafts, the 3 in-flight stories (`aep plan artifact show` on each), `epic:m1-vulkan-execution`, the decomposer report, `aep plan artifact waves --kind story` (collision lines filtered to m1 and in-flight ids), `DESIGN.md` headings and the revision line, and the three crate and workspace `Cargo.toml` files.

Surfaces: 10 placed by a cited Scope section, 0 placed only by inference (only new-file paths and the `src/vulkan` directory are inferred), 0 unplaced. I traced 20 acceptance reads to a producer in the set, and all 20 are recorded: 12 by a direct `depends_on` edge and 8 through the chain. `compile-artifact` reading `rtx3090.device.json` from `inspect-device` is one of the 8. No reader-beside-producer defect.

What I could not establish:
- Every m1 story and both in-flight observatory stories claim `CHANGELOG.md`, so `waves` gives the m1 set ten separate waves. The findings above are the collisions that remain if an operator waives `CHANGELOG.md` and `AGENTS.md` State as append-only.
- `waves` cannot know that wave `observatory-1` is already running. It places `story:m1-inspect-device`, `story:m1-kernel-plan` and `story:m1-mlp-example` in waves 3 to 5, ahead of `story:observatory-live-event-stream` in wave 6.
- Out of my lane: `story:m1-compile-artifact` and others cite `spec/domains/artifact.yaml:105`, `:146` and `:213`, and `story:m1-kernel-plan` edits that file, so the line numbers will shift. That belongs to the acceptance or design critic.
- Out of my lane: whether `story:m1-vulkan-train` (one host wait) and `story:observatory-live-event-stream` (one event per step) can both hold.

```findings
- file: .engineering/drafts/m1-checkpoint-cpu.md
  line: 55
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "this story edits crates/modelspec-runtime/src/train/runner.rs and crates/modelspec-cli/src/train.rs, the same files edited by story:m1-vulkan-train, story:m1-vulkan-profile and in-flight story:observatory-live-event-stream (all cited), and it shares cli/src/lib.rs, docs cli.rs and docs diagnostics.rs with story:m1-compile-artifact; no depends_on edge or body sentence records any of these pairs (add an ordering edge naming the files, or split the surface)"
- file: .engineering/drafts/m1-vulkan-train.md
  line: 43
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "this story edits crates/modelspec-runtime/src/train/runner.rs and crates/modelspec-cli/src/train.rs, which in-flight story:observatory-live-event-stream also edits (cited in both bodies); no edge or sentence records it (add an ordering edge naming the files, or split the surface)"
- file: .engineering/drafts/m1-vulkan-profile.md
  line: 39
  category: parallel-safety
  severity: blocker
  verdict: needs-revision
  origin: introduced
  message: "this story and story:m1-vulkan-checkpoint-infer both depend only on story:m1-vulkan-train and both edit crates/modelspec-cli/src/train.rs (cited) and the Vulkan runner loop under crates/modelspec-runtime/src/vulkan/ (inferred directory); neither body mentions the other (add an ordering edge, or split the runner module)"
- file: .engineering/drafts/m1-inspect-device.md
  line: 45
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scope omits crates/modelspec-core/Cargo.toml (inferred: DeviceProfile and its JSON form live in core and serde is added), which story:m1-kernel-plan and story:m1-compile-artifact claim; no m1 story claims Cargo.lock although five add dependencies, while in-flight story:observatory-live-event-stream claims Cargo.toml, crates/modelspec-cli/Cargo.toml and Cargo.lock and active story:cpu-matmul-row-order claims crates/modelspec-runtime/Cargo.toml, which story:m1-inspect-device, story:m1-vulkan-elementwise and story:m1-checkpoint-cpu also edit; add the missing paths and an edge or split for each pair"
- file: .engineering/drafts/m1-vulkan-elementwise.md
  line: 52
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "Taskfile.yml and AGENTS.md are claimed (cited) by unordered story:m1-mlp-example and in-flight story:observatory-live-event-stream, and .github/workflows/check.yml (inferred on the other side) and root Cargo.toml are shared with the same in-flight story; the body names none of them (add an ordering edge or split)"
- file: .engineering/drafts/m1-vulkan-reduce-matmul.md
  line: 40
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "this story edits DESIGN.md section 3.3 while in-flight story:observatory-input-adjoint edits section 3.3 pass 4 and section 4, and both bump the single revision line (DESIGN.md:4) that seven other m1 drafts also bump; no m1 story has an edge to the adjoint story (add an ordering edge naming DESIGN.md, or split the revision bump)"
- file: .engineering/drafts/m1-kernel-plan.md
  line: 68
  category: parallel-safety
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "six m1 stories claim the directory entry website/docs/reference while in-flight story:observatory-live-event-stream claims website/docs/reference/cli.md and crates.md and the directory crates/modelspec-cli/tests/; waves compares strings so it prints no collision, yet both regenerate cli.md (state the overlap or use one granularity)"
```
