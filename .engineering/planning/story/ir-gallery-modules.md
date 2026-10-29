---
format: aep.planning-md/3
id: story:ir-gallery-modules
kind: story
status: active
title: 'IR gallery modules and pipelines: every example in model, train, eval and infer views'
summary: Each example of task check-examples in all four views, collapsed and zoomable, plus a pipelines page; bert_tiny stays within 80 visible nodes per level.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:ir-gallery-atoms
- depends_on: story:diagram-pipelines
- depends_on: story:diagram-docs-blocks
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: inferred
  path: crates/modelspec-docs/src/gallery/mod.rs
- confidence: inferred
  path: crates/modelspec-docs/src/gallery/modules.rs
- confidence: inferred
  path: crates/modelspec-docs/src/gallery/pipelines.rs
- confidence: inferred
  path: website/data/diagrams
- confidence: inferred
  path: website/docs/diagrams
- confidence: inferred
  path: website/docs/gallery/modules.mdx
- confidence: inferred
  path: website/docs/gallery/pipelines.mdx
- confidence: inferred
  path: website/static/img/diagrams
revision: 8
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T21:04:23Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "proposed", to: "active", at: "2026-10-29T21:04:23Z", actor: "human:timo", revision: 8, decided_on: {"recorded":{"review_outcome":2}}}
---
# IR gallery modules and pipelines: every example, model view and train, eval and infer views

## Context

`epic:ir-svg-gallery` (Levels rows "module" and "pipeline"), DESIGN.md § 13.4 to § 13.6. Revised
2026-10-28: a module entry is a whole program of `examples/` drawn in all four views, collapsed by
default and zoomable through the program's diagram page. The pipelines page puts the training step,
the evaluation and the inference of one small program side by side. `examples/bert_tiny.mspec`
proves that a large program stays readable: 105 Tensor IR nodes and 1,132 in its AdamW training
step (`modelspec ir`, 2026-10-28).

## What it adds

- `crates/modelspec-docs/src/gallery/modules.rs`: for each file `task check-examples` checks
  (`examples/linear_regression.mspec`, `mnist_mlp.mspec`, `bert_tiny.mspec`,
  `transformer_block.mspec`), the root level of the `model`, `train`, `eval` and `infer` views on
  `website/docs/gallery/modules.mdx`, each linking to its program's diagram page.
- The diagram pages `website/docs/diagrams/<program>.mdx` of the examples that
  `story:diagram-docs-blocks` does not write (`mnist-mlp`, `transformer-block`), through the page
  writer of `crates/modelspec-docs/src/blocks.rs` that story adds; this story adds no second writer.
- `website/docs/gallery/pipelines.mdx`: the four views of `two_layer_relu.mspec`, with a short text
  per stage naming the IR it holds (DESIGN.md § 13.4).
- `CHANGELOG.md` one line.

## Order on shared files

`story:diagram-docs-blocks` lands first (a `depends_on` edge): it creates the page writer and the
diagram pages this story links to and extends. `src/gallery/mod.rs` is also edited by
`story:ir-gallery-layers` (one registration line each, merge in either order).

## Scope

- `crates/modelspec-docs/src/gallery/modules.rs`, `gallery/pipelines.rs` (new);
  `src/gallery/mod.rs` (one registration line and the index link).
- `website/docs/gallery/modules.mdx`, `pipelines.mdx`, `website/static/img/diagrams/**`,
  `website/data/diagrams/**`, `website/docs/diagrams/mnist-mlp.mdx`,
  `website/docs/diagrams/transformer-block.mdx` and the list in `website/docs/diagrams/index.md`
  (generated).
- `CHANGELOG.md` append-only.

## Acceptance

1. After `task docs-generate`, `modules.mdx` has one entry per file `task check-examples` checks,
   each with four diagrams, and `pipelines.mdx` four; `task docs-check` exits 0.
2. Test `gallery.modules.every-example` (`cargo test -p modelspec-docs`): the list of programs on
   `modules.mdx` equals the files `task check-examples` names in `Taskfile.yml`, so an example added
   there without an entry fails.
3. Test `gallery.modules.bert-tiny-levels`: every generated level of `bert_tiny`'s four views has at
   most 80 visible nodes (DESIGN.md § 13.5).
4. Test `gallery.modules.pages-and-links`: each entry of `modules.mdx` links to a
   `website/docs/diagrams/<program>.mdx` that exists after `task docs-generate`, and the
   link-resolve check of `story:diagram-docs-blocks`, run over the programs of `modules.mdx` and
   `pipelines.mdx`, passes.
5. `task website` exits 0.
6. `task ci` exits 0.

## Out of scope

- Live diagrams of a running run (`story:observatory-architecture-graph`).
- `examples/tiny_mlp.mspec`, which the M0 parser refuses (E-SYNTAX-004/005); it joins when modules
  land and `task check-examples` checks it.
