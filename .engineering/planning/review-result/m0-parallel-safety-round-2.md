---
format: aep.planning-md/3
id: review-result:m0-parallel-safety-round-2
kind: review-result
status: active
title: Parallel-safety critic, M0 decomposition round 2
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

I found no pair of stories that can run at once and share a surface without saying so. The round-1 finding is fixed.

**Round-1 finding**
- `m0-workspace-foundation` now declares `logos`, `thiserror`, `blake3` and `clap` in `[workspace.dependencies]` and per crate (`m0-workspace-foundation.md:26-33`).
- `m0-parse-subset` (`:122-123`) and `m0-tensor-ir-core` (`:185-186`) each state that they edit no `Cargo.toml` or `Cargo.lock`.
- A story that needs another crate must name it in its Lands in. No story does.

**Revisions I checked**
- `m0-cpu-sgd-training` now extends `ir/verify.rs`. It depends on `m0-reverse-mode-autodiff`, which depends on `m0-tensor-ir-core`, so all three edits to that file are ordered.
- `m0-cpu-sgd-training` also depends on `m0-lower-tensor-ir`, `m0-cpu-interpreter` and, through them, every other story. It runs last, so it cannot collide with anything.
- `cpu/interpreter.rs` is edited only by `m0-cpu-interpreter`. Both that story (`:435`) and `m0-cpu-sgd-training` (`:507`) say so.

**Pairs that can run at once**
- Pairs with no ordering path between them:
  - `m0-parse-subset` with `m0-tensor-ir-core`, `m0-cpu-interpreter` or `m0-reverse-mode-autodiff`.
  - `m0-check-types-shapes` with `m0-cpu-interpreter` or `m0-reverse-mode-autodiff`.
  - `m0-lower-tensor-ir` with `m0-cpu-interpreter` or `m0-reverse-mode-autodiff`.
- Surfaces are disjoint in every one of these pairs:
  - Source files: `syntax/*`, `resolve.rs`, `types.rs`, `hir.rs`, `lower.rs`, `passes/dce.rs`, `compile.rs`, `cpu/*`, `ops.rs`, `autodiff.rs`, `ir/training.rs`, `ir/verify.rs`.
  - CLI files: `check.rs` and `ir.rs`.
  - Test files and fixture directories: `tests/language/syntax/` and `tests/language/check/`.
  - `Taskfile.yml`, `AGENTS.md` and `DESIGN.md`, each edited on one side of a pair at most.
- `m0-check-types-shapes` reads `ops.rs` while `m0-reverse-mode-autodiff` edits it. Its body says it reads the file only, which names the sharing.
- `m0-lower-tensor-ir` says `m0-reverse-mode-autodiff` may run beside it and owns the files it leaves alone (`:310-311`).

**What I read**
- 8 stories in full, with `aep plan artifact list --kind story` and `aep plan artifact show` on each.
- `aep plan artifact graph`.
- `review-result:m0-parallel-safety-round-1`.
- The critic rubric and the parallel-safety procedure.

**Counts**
- Surfaces placed: 8 cited, 0 inferred, 0 unplaceable.
- Acceptance reads traced to a producer in the set: 16. A `depends_on` edge records 12 directly. The other 4 follow from a chain of recorded edges back to the foundation story:
  - `m0-check-types-shapes` reads `diag.rs`.
  - `m0-lower-tensor-ir` reads the `ir.rs` stub.
  - `m0-cpu-interpreter` reads the empty runtime files.
  - `m0-cpu-sgd-training` reads the foundation stubs.

**What I could not establish**
- Which story fills the `mod.rs` files (`ir/mod.rs`, `passes/mod.rs`, `cpu/mod.rs`, `train/mod.rs`) that must declare their submodules. The foundation text says only `lib.rs` declares modules. If a later story re-exports from one of them, it would collide there. Nothing in the bodies shows that, so I did not make it a finding.
- Out of my lane, for the design critic: whether `hir.rs` carries `steps`, `lr` and `seed` for `TrainConfig` in `m0-cpu-sgd-training`. If it does not, that story would edit `hir.rs` after `m0-check-types-shapes`, which is already ordered.

```findings
[]
```
