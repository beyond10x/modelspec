---
format: aep.planning-md/3
id: story:ir-gallery-atoms
kind: story
status: active
title: 'Gallery atoms and the first diagram on the site: every glyph, and two-layer-relu beside its code'
summary: 'First visible result: the gallery section with one entry per glyph, and the two-layer-relu example page showing its diagram inline beside its code, themed light and dark.'
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:ir-gallery-renderer
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: crates/modelspec-docs/Cargo.toml
- confidence: inferred
  path: crates/modelspec-docs/src/diagrams.rs
- confidence: cited
  path: crates/modelspec-docs/src/example.rs
- confidence: inferred
  path: crates/modelspec-docs/src/gallery/atoms.rs
- confidence: inferred
  path: crates/modelspec-docs/src/gallery/mod.rs
- confidence: cited
  path: crates/modelspec-docs/src/main.rs
- confidence: inferred
  path: website/data/diagrams
- confidence: cited
  path: website/docs/examples/two-layer-relu.md
- confidence: inferred
  path: website/docs/gallery
- confidence: cited
  path: website/docusaurus.config.ts
- confidence: inferred
  path: website/src/components/ModelDiagram.tsx
- confidence: inferred
  path: website/src/css/diagram.css
- confidence: inferred
  path: website/static/img/diagrams
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-29T21:04:22Z", actor: "human:timo", revision: 8}
- {from: "proposed", to: "active", at: "2026-10-29T21:04:22Z", actor: "human:timo", revision: 9}
---
# Gallery atoms and the first diagram on the site: every glyph, and two-layer-relu beside its code

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.1 and § 13.6. Revised 2026-10-28: this is the epic's first
visible result. It opens the gallery section with one entry per glyph, and shows one example
model's diagram beside its code. It also sets up everything later docs stories reuse: the
SVG output directory, the per-page data module, the site component and the theme properties.

## What it adds

- `crates/modelspec-docs/src/gallery/mod.rs` and `atoms.rs`: `website/docs/gallery/index.md` (the
  section, linking atoms now and layers, modules and pipelines as they land) and
  `website/docs/gallery/atoms.mdx`, one entry per operator glyph grouped by family: the glyph, the
  source name (or "internal" for an op no source may call), the family, and beside each forward
  operator its gradient glyph. Registered in `fn outputs`
  (`crates/modelspec-docs/src/main.rs:208`), so `task docs-check` holds it.
- `crates/modelspec-docs/src/diagrams.rs`: writes `website/static/img/diagrams/<program>/<view>/<scope-key>.svg`
  and `website/data/diagrams/<program>.json` (each level's SVG text by scope key), as
  `benchmarks.rs` writes `website/static/img/benchmarks/`.
- `website/src/components/ModelDiagram.tsx`: renders one level's SVG text inline (server-rendered,
  no client code) inside a `<figure>` with a caption and a link to the standalone file.
- `website/src/css/diagram.css` (or the site's existing custom CSS): the `--ms-*` properties under
  `:root` and `[data-theme='dark']`, mapped to Infima's colours.
- The generated example page `website/docs/examples/two-layer-relu.md` becomes `.mdx` and shows the
  model view diagram directly after its ` ```modelspec ` block (`crates/modelspec-docs/src/example.rs`).
- `CHANGELOG.md` one line.

## Scope

- `crates/modelspec-docs/src/gallery/mod.rs`, `gallery/atoms.rs`, `diagrams.rs` (new),
  `src/main.rs` (registration; cited), `src/example.rs` (cited), `Cargo.toml` (add
  `modelspec-diagram`), `Cargo.lock` (after `story:ir-gallery-renderer`, before `story:diagram-cli`,
  both ordered by `depends_on`).
- `website/docs/gallery/**`, `website/static/img/diagrams/**`, `website/data/diagrams/**`
  (generated); `website/docs/examples/two-layer-relu.mdx` (generated, replaces `.md`).
- `website/src/components/ModelDiagram.tsx`, `website/src/css/diagram.css` (new),
  `website/docusaurus.config.ts` (the CSS, if the product layer takes no custom CSS otherwise).
- `CHANGELOG.md` append-only.

## Acceptance

1. `task docs-check` exits 0 after `task docs-generate`, and fails when one byte of
   `website/static/img/diagrams/two-layer-relu/model/FitTwoLayer.svg` or of
   `website/docs/gallery/atoms.mdx` is changed (test `gallery.atoms.drift`, `cargo test -p
   modelspec-docs`).
2. `gallery.atoms.every-glyph`: `atoms.mdx` has one entry per `<symbol id="ms-op-…">` of
   `modelspec_diagram::glyph::sprite()`, and each forward operator's entry shows its gradient glyph
   where `ops::gradient` has one.
3. `gallery.example.two-layer-relu`: in `website/docs/examples/two-layer-relu.mdx` the first element
   after the ` ```modelspec ` block is the `ModelDiagram` of `two-layer-relu`, model view.
4. `task website` exits 0, and the built page `website/build/docs/examples/two-layer-relu.html`
   (the site sets `trailingSlash: false`) holds the diagram as an inline `<svg role="img"`, not an
   `<img>`, with a `data-op="MatMul"` element in it: the command shown, run once by the implementor
   and its output pasted into the story's evidence.
5. `task ci` exits 0.

## Out of scope

- Diagrams beside the other blocks of the site (`story:diagram-docs-blocks`).
- Zoom links and composite nodes (`story:diagram-zoom`).
