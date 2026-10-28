---
format: aep.planning-md/3
id: story:diagram-docs-blocks
kind: story
status: draft
title: A diagram beside every modelspec program of the documentation, and one diagram page per program
summary: A manifest of all 12 modelspec blocks (whole, fragment of a fixture, or skipped with its code), a diagram beside each whole program, and one zoomable diagram page per program.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:ir-gallery-atoms
- depends_on: story:diagram-pipelines
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: README.md
- confidence: inferred
  path: crates/modelspec-docs/src/blocks.rs
- confidence: inferred
  path: crates/modelspec-docs/src/diagrams.rs
- confidence: cited
  path: crates/modelspec-docs/src/example.rs
- confidence: cited
  path: crates/modelspec-docs/src/main.rs
- confidence: inferred
  path: website/data/diagrams
- confidence: cited
  path: website/docs/concepts/bert-operators.md
- confidence: cited
  path: website/docs/concepts/language.md
- confidence: inferred
  path: website/docs/diagrams
- confidence: cited
  path: website/docs/examples
- confidence: cited
  path: website/docs/guides/write-a-model.md
- confidence: cited
  path: website/docs/index.md
- confidence: cited
  path: website/sidebars.ts
- confidence: inferred
  path: website/static/img/diagrams
revision: 7
---
# A diagram beside every modelspec program of the documentation, and one diagram page per program

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.6. The operator asked that "anywhere in the repo where in the
docs we render code of model spec notation" the diagram stands beside it. The repository holds 14
` ```modelspec ` blocks outside the generated gallery (counted on 2026-10-28): 12 under
`website/docs`, one in `README.md` and one in `DESIGN.md`. 7 are whole programs, 3 are fragments of
one program, and 4 are blocks the compiler refuses on purpose.

| file | block | disposition |
|---|---|---|
| `website/docs/index.md` | 1 | whole (`LinearReg`) |
| `website/docs/concepts/language.md` | 1 | whole (`TwoLayer`) |
| `website/docs/concepts/language.md` | 2 | skipped: target syntax, E-SYNTAX-004 (`module`) |
| `website/docs/guides/write-a-model.md` | 1, 2, 3 | fragments of `crates/modelspec-docs/models/two_layer_relu.mspec`, one diagram after block 3 |
| `website/docs/concepts/bert-operators.md` | 1 | whole (`crates/modelspec-cli/tests/fixtures/mlm_bag.mspec`) |
| `website/docs/examples/linear-regression.md` | 1 | whole (generated page) |
| `website/docs/examples/two-layer-relu.md` | 1 | whole (generated page; its diagram is written by `story:ir-gallery-atoms` and only checked here) |
| `website/docs/examples/bert-tiny.md` | 1 | whole (generated page) |
| `website/docs/examples/beyond-m0.md` | 1 | skipped: target syntax, E-SYNTAX-004 and E-SYNTAX-005 |
| `website/docs/examples/shape-error.md` | 1 | skipped: the refusal is the page's subject |
| `README.md` | 1 (line 19) | whole (`LinearReg`), drawn as a Markdown image |
| `DESIGN.md` | 1 (line 51) | skipped: target syntax, E-SYNTAX-004 (`module`) |

The generated gallery pages under `website/docs/gallery/` are not in the manifest: each gallery
story writes its pages with their diagrams already beside every program it shows.

## What it adds

- `crates/modelspec-docs/src/blocks.rs`: the manifest above as data; a scan of every
  ` ```modelspec ` fence in `README.md`, `DESIGN.md` and `website/docs/**/*.md` and `*.mdx` outside
  `website/docs/gallery/`; for each whole block and fragment set, the program's levels (model view
  and `train` view) written through `diagrams.rs` (from `story:ir-gallery-atoms`).
- One page per program, `website/docs/diagrams/<program>.mdx`, every level of its `model` and
  `train` views under its own heading, anchored by scope key; composite links point at those
  anchors. `website/docs/diagrams/index.md` lists the programs. The page writer is a function of
  `blocks.rs` that `story:ir-gallery-modules` calls for the examples it adds.
- The hand-written pages with a whole block or a fragment set become `.mdx` (`index`,
  `concepts/language`, `concepts/bert-operators`, `guides/write-a-model`; their URLs do not change)
  and get a `<ModelDiagram>` after the block. The generated example pages get theirs from
  `example.rs`.
- `README.md`: after the block, one Markdown image of the committed standalone SVG
  `website/static/img/diagrams/linear-reg/model/<root>.svg` (it carries its own glyphs and the
  theme's fallback colours, so GitHub shows it without the site), linking to the program's diagram
  page on the site.

## Scope

- `crates/modelspec-docs/src/blocks.rs` (new), `src/main.rs`, `src/example.rs`, `src/diagrams.rs`
  (cited).
- `website/docs/index.mdx`, `concepts/language.mdx`, `concepts/bert-operators.mdx`,
  `guides/write-a-model.mdx` (renamed from `.md`, hand-written: one import and one element each).
- `README.md` (one image line after the block; cited).
- `website/docs/diagrams/**`, `website/static/img/diagrams/**`, `website/data/diagrams/**`,
  `website/docs/examples/*.mdx` (generated). `website/sidebars.ts` if the section is not
  autogenerated.
- `CHANGELOG.md` one line.

## Acceptance

Tests in `crates/modelspec-docs` (`cargo test -p modelspec-docs`, `task ci`):

1. `docs.blocks.every-block-listed`: every ` ```modelspec ` fence in `README.md`, `DESIGN.md` and
   under `website/docs` outside `website/docs/gallery/` has a manifest entry, and every entry names
   a fence that exists; adding an unlisted block to any of them fails with its file and line.
2. `docs.blocks.whole-has-diagram`: after each whole block and after the last block of each
   fragment set, the next element before any heading is the `ModelDiagram` of that program, and in
   `README.md` the next non-blank line is the image of that program's SVG, a file that exists.
3. `docs.blocks.fragments-in-fixture`: each fragment's text, whitespace-normalized, is contained in
   its fixture.
4. `docs.blocks.skips-still-refused`: each skipped block, compiled, still answers the code its entry
   names; a skipped block that compiles fails the test, naming the block, so its diagram is added
   when modules land.
5. `docs.diagrams.links-resolve`: every `href` in every SVG generated for a program of the manifest
   names a heading anchor that exists on the generated diagram page it points at. The check is a
   function over a list of programs, which `story:ir-gallery-modules` calls for its own.
6. `task docs-check` exits 0, and `task website` exits 0 (Docusaurus `onBrokenLinks` and
   `onBrokenAnchors` are `throw`).

## Out of scope

- The gallery pages (`story:ir-gallery-atoms`, `story:ir-gallery-layers`,
  `story:ir-gallery-modules`).
- Diagrams beside `.mspec` files that no documentation page shows.
