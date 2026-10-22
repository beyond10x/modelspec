---
format: aep.planning-md/3
id: story:ir-gallery-modules
kind: story
status: draft
title: 'IR gallery modules: each example model, forward and training step'
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:ir-gallery-atoms
revision: 1
---
# IR gallery modules: each example model, forward and training step

## Context

`epic:ir-svg-gallery` (Levels table row "module"): a module entry shows a whole model of
`examples/`, its forward graph, and the training step (forward, backward, update). The compiler
already produces both: `modelspec_core::compile::compile_training`
(`crates/modelspec-core/src/compile.rs:54`) returns `CompiledTraining { forward, training }`, the
Tensor IR with its loss (what `modelspec ir --stage tensor` prints) and the verified `TrainingStep`
with SGD attached and every value tagged forward, loss, gradient or optimizer (what
`modelspec ir --stage training` prints). This story renders both for each example.

`examples/` holds `linear_regression.mspec`, `mnist_mlp.mspec` and `tiny_mlp.mspec`. The first two
are in the M0 subset; `tiny_mlp.mspec` is target syntax the M0 parser refuses until DESIGN.md
section 11 iteration 3 (AGENTS.md, Rules). Its entry says so and names the first diagnostic code,
and it gets no diagram.

## What it adds

- `crates/modelspec-docs/src/gallery/modules.rs`: a fixed list of module entries, one per file of
  `examples/`, each stating whether it must compile. Generation fails when a file of `examples/` has
  no entry, when an entry names a missing file, when a must-compile entry does not compile, and when
  a must-refuse entry compiles (so the page is updated when `tiny_mlp.mspec` starts to check).
- Per compiling example `<stem>`: `module-<stem>.svg` from `CompiledTraining.forward.program`, and
  `module-<stem>-training.svg` from `CompiledTraining.training.training` with its subgraph tags, so
  forward, loss, gradient and optimizer nodes are told apart.
- `website/docs/gallery/modules.md` (generated): per example a link to its source on GitHub, the
  typed inputs, parameters and loss, both diagrams, and for a refused example the diagnostic line;
  `index.md` links it. Each module links its example page under `website/docs/examples/` where one
  exists (`linear-regression.md`).
- `CHANGELOG.md`: one line.

## Domain relations

- Module entry to example file: one entry per file of `examples/`, exhaustively. Inferred from the
  `examples/` directory and the rule in AGENTS.md that every example becomes part of the gate once
  the parser supports it; no ess/1 document declares it.
- Training diagram tags from `TrainingProgram.subgraphs`. Inferred from
  `crates/modelspec-core/src/ir/training.rs:33` and `:66`; no ess/1 relation declares it.

## Scope

- `crates/modelspec-docs/src/gallery/modules.rs` (new; inferred);
  `crates/modelspec-docs/src/gallery/mod.rs` (one registration line and the `index.md` link; also
  edited by `story:ir-gallery-layers`, one line each, merge in either order).
- `website/docs/gallery/modules.md`, `module-*.svg` (generated).
- `CHANGELOG.md` (append-only).
- Reads, does not edit: `examples/*.mspec`, `crates/modelspec-cli/**`.

## Acceptance

1. After `task docs-generate`, `website/docs/gallery/` also holds `modules.md`,
   `module-linear_regression.svg`, `module-linear_regression-training.svg`, `module-mnist_mlp.svg`
   and `module-mnist_mlp-training.svg`, and no SVG for `tiny_mlp`; `task docs-check` exits 0.
2. Test `gallery.modules.matches-ir` (`cargo test -p modelspec-docs`): for each compiling example,
   the node count of `module-<stem>.svg` equals the number of definitions
   `modelspec ir <file> --stage tensor` prints, and that of `module-<stem>-training.svg` equals the
   number `modelspec ir <file> --stage training` prints, each node `data-op` and `data-shape` equal
   to its line.
3. Test `gallery.modules.training-subgraphs`: in `module-linear_regression-training.svg` and
   `module-mnist_mlp-training.svg` each of `data-subgraph="forward"`, `"loss"`, `"gradient"` and
   `"optimizer"` occurs, and the optimizer nodes number one `Scalar` for the learning rate plus a
   `Mul` and a `Sub` per parameter.
4. Test `gallery.modules.every-example`: against a scratch repository root with an extra
   `examples/new.mspec`, gallery generation returns an error naming `examples/new.mspec`.
5. `modules.md` holds the diagnostic code that `modelspec check examples/tiny_mlp.mspec` prints
   first, checked by test `gallery.modules.refused-entry`.
6. `task ci` exits 0.

## Out of scope

- Effect tokens and state versions drawn as edges; the training diagram shows values and their
  operands only.
- Kernel IR and the execution plan (epic, Out of scope).
- Models outside `examples/` (`crates/modelspec-docs/models/`); `two_layer_relu.mspec` has its
  example page already.
- The live architecture view of `story:observatory-architecture-graph`, which draws a `ModelGraph`
  in the observatory UI; this story shares no code with it.
