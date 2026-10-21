---
format: aep.planning-md/3
id: story:bert-tiny-cpu-model
kind: story
status: draft
title: BERT-tiny as a checked example, and its hidden-32 twin gradient-checked and trained on the CPU
summary: examples/bert_tiny.mspec (4,367,290 parameters) joins check-examples; a hidden-32 twin passes a whole-model f64 gradient check and trains to identical loss curves from one seed.
relations:
- decomposes: epic:bert-from-scratch
- serves: vision:modelspec
- depends_on: story:bert-heldout-evaluation
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Taskfile.yml
- confidence: cited
  path: crates/modelspec-cli/tests/bert_cpu_train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/bert_tiny.rs
- confidence: cited
  path: crates/modelspec-cli/tests/fixtures/bert_cpu.mspec
- confidence: cited
  path: crates/modelspec-docs/src/example.rs
- confidence: cited
  path: crates/modelspec-docs/src/status.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/gradients_bert_cpu.rs
- confidence: cited
  path: examples/bert_tiny.mspec
- confidence: cited
  path: website/data/status.json
- confidence: cited
  path: website/docs/examples/
revision: 3
---
# BERT-tiny as a checked example, and its hidden-32 twin gradient-checked and trained on the CPU

## Context

`epic:bert-from-scratch` (In scope, Model size: "BERT-tiny class first (2 layers, hidden 128, 2
heads, vocabulary about 30k, sequence 128; about 4.4M parameters, to verify against Turc et al.
2019, arXiv 1908.08962)"; Execution: "the CPU interpreter for gradient checks on a 2-layer,
hidden-32 configuration"; Acceptance: "A BERT-tiny `.mspec` passes `modelspec check`", "Two runs
with one seed produce identical loss curves on the CPU configuration"). The operators exist after
`story:bert-adamw-warmup`; the data path after `story:bert-heldout-evaluation`. Modules and
generics are DESIGN.md § 11 iteration 3 and are not needed: the two layers are written out, as the
M0 parser reads one `model` with `param` and `fn` items (`crates/modelspec-core/src/syntax/parser.rs`,
module doc grammar).

## What this story decides

- `examples/bert_tiny.mspec`: `L = 2`, `H = 128`, `A = 2`, feed-forward width `F = 512`
  (Turc et al. 2019), `S = 128`, `V = 30522` (the vocabulary size of the original BERT release, in
  the "about 30k" of the epic), batch `B = 32`. Post-layer-norm encoder layers as in Devlin et al.
  2019: attention, `add`, `layer_norm`, then `gelu` feed-forward, `add`, `layer_norm`; token and
  learned positional embeddings followed by `layer_norm`; an MLM head of a dense layer, `gelu`,
  `layer_norm` and a decoder tied to the token table with its own bias; `dropout` with
  `p = 0.1` after the embeddings and each sublayer; loss `cross_entropy` over `labels` and
  `weights`; `adamw(...)` with the schedule `story:bert-adamw-warmup` settled.
- No token-type embedding and no pooler, because the epic trains MLM only. The parameter count is
  then `V H + S H + 2H + L (4 (H^2 + H) + 2 (H F) + F + H + 4H) + (H^2 + H) + 2H + V`
  = 4,367,290, the "about 4.4M parameters" of the epic.
- `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec`: the same file with `H = 32`, `A = 2`,
  `F = 64`, `S = 16`, `B = 2` and `V = 300`, the vocabulary of the fixture tokenizer of
  `story:bert-tokenizer`.
- `bert_tiny.mspec` joins `task check-examples` (AGENTS.md Rules: an example joins the gate once
  the parser supports its constructs) and the example pages of the documentation site.

## Order on shared files

This story edits `Taskfile.yml`, `AGENTS.md`, `crates/modelspec-docs/src/example.rs`,
`crates/modelspec-docs/src/status.rs`, `website/` generated pages and `CHANGELOG.md` after
`story:bert-heldout-evaluation` and the stories before it, so it depends on that story.
`story:bert-vulkan-kernels-rowwise` follows it.

## Domain relations

- Every parameter of the two files is a `modelspec.artifact.ParameterSpec` of
  `CompiledArtifact.parameters`, unique by `path` and by `canonical_parameter_id`
  (`spec/domains/artifact.yaml` invariants); the tied decoder reads the token table and declares
  no second parameter for it (the summary of `parameters` there: "no two parameters alias").

## Scope

- `examples/bert_tiny.mspec` (new); `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec` (new).
- `crates/modelspec-runtime/tests/gradients_bert_cpu.rs` (new), on the harness of
  `crates/modelspec-runtime/tests/common/mod.rs`; `crates/modelspec-cli/tests/bert_tiny.rs` and
  `bert_cpu_train.rs` (new).
- `Taskfile.yml` (`check-examples`), `AGENTS.md` (Commands, State, the examples rule),
  `crates/modelspec-docs/src/example.rs` (an `EXAMPLES` entry) and `src/status.rs` (a
  `CAPABILITIES` entry); `website/docs/examples/` and `website/data/status.json` through
  `task docs-generate`; `CHANGELOG.md` one line.

## Acceptance

`modelspec check examples/bert_tiny.mspec` exits 0 with 4,367,290 parameters, and the hidden-32
twin passes an `f64` finite-difference check of every parameter and trains to identical loss curves
twice from one seed on the CPU.

Scenarios, each run by `task ci` through the command shown:

1. `task check-examples` runs `modelspec check examples/bert_tiny.mspec` and exits 0.
2. `cargo test -p modelspec-cli --test bert_tiny parameter_count`: the parameter types that
   `modelspec ir examples/bert_tiny.mspec --stage tensor` prints have 4,367,290 elements in total,
   and the path of every parameter is unique.
3. `cargo test -p modelspec-runtime --test gradients_bert_cpu`: for one fixture batch of
   `bert_cpu.mspec` and fixed dropout tokens, 16 elements of every parameter (all of a parameter
   with fewer), chosen by a fixed test seed, have gradients within `TOL_F64` of central
   differences in `f64`; the tied token table is among them.
4. `cargo test -p modelspec-cli --test bert_cpu_train reproducible`: `modelspec train
   --backend cpu --data` on `bert_cpu.mspec` and `--eval-every`, with data the test packs as
   `train_data trains` of `story:bert-train-data-binding` does (the corpus fixture of
   `story:bert-corpus-splits`, `--vocab-size 300`, `--seq-len 16`, a temporary directory under
   `target/`), run
   twice with one seed, prints identical stdout, `step` and `eval` lines included, and its last
   `step` loss is below its step-0 loss.
5. `task docs-check` exits 0 with `website/docs/examples/` holding a BERT-tiny page and
   `website/data/status.json` a BERT-tiny CPU entry whose evidence is the test of scenario 4.

## Out of scope

- Training BERT-tiny itself (`story:bert-tiny-gpu-pretraining`, on the GPU).
- BERT-mini and BERT-small (optional in the epic; not drafted).
