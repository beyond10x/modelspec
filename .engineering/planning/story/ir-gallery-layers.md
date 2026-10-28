---
format: aep.planning-md/3
id: story:ir-gallery-layers
kind: story
status: draft
title: 'IR gallery layers: composed blocks as zoomable function scopes'
summary: Small models whose block is a function scope, drawn collapsed as one composite and expanded to its operators, including the transformer blocks now in the registry.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:ir-gallery-atoms
- depends_on: story:diagram-zoom
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: inferred
  path: crates/modelspec-docs/models/layers
- confidence: inferred
  path: crates/modelspec-docs/src/gallery/layers.rs
- confidence: inferred
  path: crates/modelspec-docs/src/gallery/mod.rs
- confidence: inferred
  path: website/data/diagrams
- confidence: inferred
  path: website/docs/gallery/layers.mdx
- confidence: inferred
  path: website/static/img/diagrams
revision: 5
---
# IR gallery layers: composed blocks as zoomable function scopes

## Context

`epic:ir-svg-gallery` (Levels row "layer"), DESIGN.md § 13.3. Revised 2026-10-28: a layer entry is now
a small model whose block is a model function, so its diagram has a Composite node that zooms into
the block, instead of a flat forward slice. The language has no `layer` construct yet (E-SYNTAX-005,
DESIGN.md § 11 iteration 3), and a model function is the composition unit today (DESIGN.md § 2.1).
Until `story:decider-model-function-calls` lets one function call another, each entry's block is its
model's `forward`, inlined as the Function scope `forward`; after it, an entry may be a `fn` that
`forward` calls, and the acceptance below holds unchanged. Since the first draft of this
story the registry gained `layer_norm`, `softmax`, `attention`, `gelu` and `add_rows`
(`crates/modelspec-core/src/resolve.rs:64`), so the transformer blocks join the list.

| entry | block |
|---|---|
| `dense` | `matmul(x, w)` |
| `dense-relu` | `relu(matmul(x, w))` |
| `dense-bias-gelu` | `gelu(add_rows(matmul(x, w), b))` |
| `layer-norm-residual` | `layer_norm(add(x, f), gamma, beta)` |
| `self-attention` | `attention(q, k, v, key_mask, heads = 2)` over three projections |

The list is an editorial choice taken here and reversible.

## What it adds

- `crates/modelspec-docs/models/layers/*.mspec` (new), one per entry, in the language as it is,
  `device = cpu`.
- `crates/modelspec-docs/src/gallery/layers.rs`: compiles each, fails generation on any diagnostic,
  writes the model view's levels through `diagrams.rs`; `website/docs/gallery/layers.mdx`: per
  entry its source, the collapsed diagram (one Composite for the block) and the expanded one.
- `CHANGELOG.md` one line.

## Scope

- `crates/modelspec-docs/models/layers/*.mspec`, `src/gallery/layers.rs` (new);
  `src/gallery/mod.rs` (one registration line and the index link; also edited by
  `story:ir-gallery-modules`, one line each, merge in either order).
- `website/docs/gallery/layers.mdx`, `website/static/img/diagrams/layer-*/**`,
  `website/data/diagrams/layer-*.json` (generated).
- `CHANGELOG.md` append-only.

## Acceptance

1. After `task docs-generate`, `website/docs/gallery/layers.mdx` and one diagram set per entry
   exist; `task docs-check` exits 0.
2. `modelspec check` exits 0 on each file under `crates/modelspec-docs/models/layers/`.
3. Test `gallery.layers.block-is-composite` (`cargo test -p modelspec-docs`): in each entry's
   collapsed level the block is exactly one Composite node with glyph `scope-function` and one In
   port per argument; in its expanded level the Operator nodes are exactly the block's operators (for
   `dense-bias-gelu`: `MatMul`, `AddRows`, `Gelu`, with the parameters as chips).
4. Test `gallery.layers.refuses-broken-model`: a layer model with a shape error makes generation
   return an error naming the file and its diagnostic code, not a page.
5. `task ci` exits 0.

## Out of scope

- A `layer` or `module` construct (DESIGN.md § 11 iteration 3); when it lands, these entries become
  layer instances with `scope-layer` glyphs.
