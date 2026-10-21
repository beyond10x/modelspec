---
format: aep.planning-md/3
id: review-result:bert-parallel-safety-round-1
kind: review-result
status: active
title: Plan critic parallel-safety, epic:bert-from-scratch, round 1
relations:
- reviews: epic:bert-from-scratch
- reviews: story:bert-adamw-warmup
- reviews: story:bert-attention
- reviews: story:bert-corpus-splits
- reviews: story:bert-dense-gelu
- reviews: story:bert-dropout
- reviews: story:bert-embedding-lookup
- reviews: story:bert-heldout-evaluation
- reviews: story:bert-layer-norm-softmax
- reviews: story:bert-mlm-batches
- reviews: story:bert-mlm-cross-entropy
- reviews: story:bert-tiny-cpu-model
- reviews: story:bert-tiny-gpu-pretraining
- reviews: story:bert-tokenizer
- reviews: story:bert-train-data-binding
- reviews: story:bert-vulkan-kernels-attention-loss-adamw
- reviews: story:bert-vulkan-kernels-rowwise
revision: 1
---
approve

What I read: 16 `story:bert-*` bodies in full, plus 8 observatory story bodies for the shared-surface check; `aep plan artifact list`, `graph`, `waves`, `validate` (valid), `show`; `Cargo.toml`, `crates/modelspec-docs/src/{cli,diagnostics,status}.rs` and the `crates/*/tests` directories.

Surfaces: all 16 cited from their bodies, 0 inferred, 0 unplaceable. Every body has an "Order on shared files" section and a Scope section with paths.

Reads and producers: 24 acceptance reads traced to a producer inside the set or an observatory ancestor; 9 on a direct `depends_on` edge, 15 on a transitive path, none unrecorded.

Why the set is parallel-safe:
- The graph is one chain: saliency → embedding → dense-gelu → layer-norm-softmax → attention → mlm-cross-entropy → dropout → adamw-warmup → mlm-batches → train-data-binding → heldout-evaluation → tiny-cpu-model → vulkan-rowwise → vulkan-attention-loss-adamw → tiny-gpu-pretraining. Every edge matches its body.
- The only concurrent lane is `bert-corpus-splits` then `bert-tokenizer`, beside the seven operator stories, and the two sides share no file. The lane edits `crates/modelspec-cli/src/lib.rs`, `Taskfile.yml`, `AGENTS.md`, runtime `Cargo.toml` and `Cargo.lock`, `spec/domains/data.yaml` and `cli.md`; the operator stories edit none of these. `blake3` is already in `crates/modelspec-runtime/Cargo.toml:16`. `cli.md` is generated from clap; the operator stories touch `operators.md` and `diagnostics.md`.

Surfaces shared with the observatory stories: the bert chain is ordered after all 8, because `bert-embedding-lookup` and `bert-corpus-splits` depend on `story:observatory-saliency-gradient-x-input`, and every other observatory story is its ancestor.
- Operator stories share `spec/domains/observatory.yaml` (`TensorOperator`), `crates/modelspec-observatory/`, `ui/src/generated/`, `autodiff.rs`, `DESIGN.md` § 3.3 and § 4, `CHANGELOG.md`.
- `bert-adamw-warmup`, `bert-dropout`, `bert-train-data-binding`, `bert-heldout-evaluation`, `bert-vulkan-kernels-attention-loss-adamw` share `runner.rs`, `optimizer.rs`, `crates/modelspec-cli/src/train.rs`.
- `bert-corpus-splits` shares `Taskfile.yml`, `AGENTS.md`, `Cargo.toml`, `Cargo.lock`, `crates/modelspec-cli/src/lib.rs`, `cli.md`, and the schema drift check from `story:observatory-live-event-stream`.

What I could not establish:
- `aep plan artifact waves` lists all 16 as `unassessed`: no typed `scope` is recorded. Relayed, not counted as a finding.
- Whether `ess generate --kind schema --check` over `crates/modelspec-runtime/schema/` also drifts when the operator stories edit `observatory.yaml`; not run.
- Out of lane: `bert-tiny-cpu-model` scenario 4's fixture packed data has only a temporary producer; DESIGN.md revision-line greps hold only until the next bump; the operator and kernel path may be more serial than needed.

```findings
[]
```
