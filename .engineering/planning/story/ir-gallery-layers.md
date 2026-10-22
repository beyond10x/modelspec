---
format: aep.planning-md/3
id: story:ir-gallery-layers
kind: story
status: draft
title: 'IR gallery layers: composed blocks as subgraphs'
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:ir-gallery-atoms
revision: 1
---
# IR gallery layers: composed blocks as subgraphs

## Context

`epic:ir-svg-gallery` (Levels table row "layer"): a layer entry shows a composed block, such as a
dense layer with its activation, as a subgraph. DESIGN.md section 0 invariant 2 says composition
does not mandate kernel boundaries and functions or layers are inlined, and the M0 language has no
`layer` construct (the parser refuses it, `website/docs/examples/beyond-m0.md`). So a layer has no
node of its own in the Tensor IR; an entry is the inlined IR of a small model whose forward is that
block. The M0 parser requires a `train` block in every file (`HirTrain` is not optional,
`crates/modelspec-core/src/hir.rs:16`, inferred), so each layer model is a complete M0 file, and
the diagram keeps only the forward slice: the nodes the `prediction` value
(`LoweredProgram.prediction`, `crates/modelspec-core/src/lower.rs:36`) depends on, without the loss.

The list of layers is an editorial choice taken here and reversible: the blocks the M0 operator set
can express and the examples use.

| entry | forward |
|---|---|
| `dense` | `matmul(x, w)` |
| `dense-relu` | `relu(matmul(x, w))` |
| `dense-bias-relu` | `relu(add(matmul(x, w), b))`, `b` of the full output shape, since `add` takes equal shapes or a rank-0 operand (`crates/modelspec-docs/src/operators.rs:28`, the `add` examples) |

## What it adds

- `crates/modelspec-docs/models/layers/dense.mspec`, `dense_relu.mspec`, `dense_bias_relu.mspec`
  (new), each in the M0 subset with `device = cpu`.
- `crates/modelspec-docs/src/gallery/layers.rs`: compiles each model with
  `modelspec_core::compile::compile_tensor` (`crates/modelspec-core/src/compile.rs:136`), fails
  generation on any diagnostic, takes the forward slice and renders `layer-<entry>.svg`.
- `website/docs/gallery/layers.md` (generated): per layer its source in a fenced block, the typed
  inputs and output, and the diagram; `index.md` links it.
- `CHANGELOG.md`: one line.

## Domain relations

- Layer entry to Tensor IR: the forward slice of one model, one diagram node per `TensorNode` the
  prediction reaches. Inferred from `crates/modelspec-core/src/lower.rs:31` (`LoweredProgram {
  program, prediction }`) and DESIGN.md section 0 invariant 2 (layers inlined); no ess/1 relation
  declares it.

## Scope

- `crates/modelspec-docs/models/layers/*.mspec` (new; inferred).
- `crates/modelspec-docs/src/gallery/layers.rs` (new; inferred); `crates/modelspec-docs/src/gallery/mod.rs`
  (one registration line and the `index.md` link; also edited by `story:ir-gallery-modules`, one
  line each, merge in either order).
- `website/docs/gallery/layers.md` and `layer-*.svg` (generated).
- `CHANGELOG.md` (append-only).

## Acceptance

1. After `task docs-generate`, `website/docs/gallery/` also holds `layers.md`,
   `layer-dense.svg`, `layer-dense-relu.svg` and `layer-dense-bias-relu.svg`; `task docs-check`
   exits 0.
2. `modelspec check` exits 0 on each of the three files under `crates/modelspec-docs/models/layers/`.
3. Test `gallery.layers.forward-slice` (`cargo test -p modelspec-docs`): `layer-dense-relu.svg`
   has exactly the nodes `Input`, `ParamRead`, `MatMul`, `Relu`, and no `Mse` node and no node of
   the target input; `layer-dense-bias-relu.svg` has one `Add` node whose operands are the `MatMul`
   and a `ParamRead`.
4. Test `gallery.layers.refuses-broken-model`: a layer model with a shape error makes the gallery
   generation return an error naming the file and its diagnostic code, not a page.
5. `task ci` exits 0.

## Out of scope

- A `layer` or `module` construct in the language (DESIGN.md section 11 iteration 3).
- Layers the M0 operators cannot express (softmax, layer norm, attention); they join when their
  operators enter the registry.
- Grouping boxes around a layer inside a module diagram.
