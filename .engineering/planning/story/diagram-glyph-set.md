---
format: aep.planning-md/3
id: story:diagram-glyph-set
kind: story
status: active
title: 'Glyph set: one SVG symbol per operator, in families, themable'
summary: New crate modelspec-diagram with one generated SVG symbol per TensorOp in eleven families, gradient, stage, scope and endpoint glyphs, themed by CSS custom properties.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
scope:
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: Cargo.toml
- confidence: inferred
  path: crates/modelspec-diagram/Cargo.toml
- confidence: inferred
  path: crates/modelspec-diagram/src/glyph.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/lib.rs
- confidence: inferred
  path: crates/modelspec-diagram/tests/glyphs.rs
revision: 6
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T19:01:26Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"review_outcome":1}}}
- {from: "proposed", to: "active", at: "2026-10-28T19:01:26Z", actor: "human:timo", revision: 6, decided_on: {"recorded":{"review_outcome":1}}}
---
# Glyph set: one SVG symbol per operator, in families, themable

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.1. Every Tensor IR operation gets a glyph, generated from Rust,
themable through CSS custom properties. The registry is the source: the operators a file may call
are `resolve::OPERATORS` (`crates/modelspec-core/src/resolve.rs:64`), the operations the IR holds
are `TensorOp` (`crates/modelspec-core/src/ir/tensor.rs:129`, names at `:249`), and the ESS enum
`modelspec.observatory.TensorOperator` names them (pinned by the test at
`crates/modelspec-core/src/ir/tensor.rs:800`). This story creates the crate the whole epic draws
with and needs nothing else of it.

## What it adds

- `crates/modelspec-diagram/` (new workspace member): `Cargo.toml` depending on `modelspec-core` and
  `serde` (workspace), with an optional `ts-rs` (workspace) behind a feature `ts`; `src/lib.rs`;
  `src/glyph.rs`.
- The domain enums the glyphs are named by, owned by this story in `src/glyph.rs`: `GlyphId`
  (newtype of `String`), `OperatorFamily`, `PipelineStage` and `EndpointRole`, as
  `spec/domains/diagram.yaml` declares them, each deriving `Serialize` and `Deserialize` with the
  domain's variant names and `#[cfg_attr(feature = "ts", derive(ts_rs::TS))]`.
  `story:ir-gallery-renderer` re-exports them from `src/model.rs` and defines no second copy.
- `InputRole` (Rust only, not a domain type, because the `GlyphId` it selects is what the domain
  carries): `Data`, `DropoutToken`, `HostStep`. The graph builder classifies an `Input` by
  `dropout_sites` (`crates/modelspec-core/src/ir/training.rs:550`) and `OptimizerStep::host_inputs`
  (`:501`); every other node passes `Data`.
- `glyph_of(op: &TensorOp, role: InputRole) -> GlyphId`: an exhaustive `match`, no wildcard arm; an
  `Input` gives `op-input`, `op-input-token` or `op-input-step` by role, and the role is ignored for
  every other op. `family_of(op) -> OperatorFamily` likewise; the ids and families of DESIGN.md
  § 13.1, gradient ops their forward glyph with the dashed outline and the `∇` mark.
- The non-operator glyphs: `scope-function`, `scope-adjoint`, `scope-update-sgd`,
  `scope-update-adamw`, `scope-update-muon`, `scope-module`, `scope-layer`, one `stage-<stage>` per
  `PipelineStage` (six) and one `endpoint-<role>` per `EndpointRole` (four).
- `sprite() -> String`: one `<svg>` holding a `<symbol id="ms-<id>" viewBox="0 0 24 24">` per glyph
  in id order and the `<style>` with the custom properties `--ms-diagram-ink`,
  `--ms-diagram-paper`, `--ms-diagram-muted`, `--ms-family-<family>`, `--ms-subgraph-<subgraph>`,
  light defaults and `@media (prefers-color-scheme: dark)` defaults.

## Scope

- `Cargo.toml` (workspace `members`, one entry; cited), `Cargo.lock`.
- `crates/modelspec-diagram/Cargo.toml`, `src/lib.rs`, `src/glyph.rs`, `tests/glyphs.rs` (new).
- Not `crates/modelspec-core/**`.

## Acceptance

Tests in `crates/modelspec-diagram/tests/glyphs.rs`, run by `cargo test -p modelspec-diagram` and so
by `task ci`:

1. `glyph.every-operator`: every name in `resolve::OPERATORS` and every variant of
   `modelspec.observatory.TensorOperator` (read from its generated JSON Schema, as the test at
   `tensor.rs:800` reads it) has exactly one `<symbol id="ms-op-…">` in `sprite()`, and no two
   operators share an id.
2. `glyph.gradient-mark`: each internal gradient op's symbol (`ReluGrad`, `EmbeddingGrad`,
   `GeluGrad`, `LayerNormGradX`, `LayerNormGradGamma`, `SoftmaxGrad`, `AttentionGradQ`,
   `AttentionGradK`, `AttentionGradV`, `CrossEntropyGrad`, `DropoutGrad`) contains the gradient
   mark and the outline of its forward operator's family.
3. `glyph.families`: every `OperatorFamily` variant has at least one operator glyph and a
   `--ms-family-<family>` property; every stage, endpoint and scope glyph of DESIGN.md § 13.1 is
   present, and `op-input-token` and `op-input-step` are what `glyph_of(Input, DropoutToken)` and
   `glyph_of(Input, HostStep)` return.
4. `glyph.themable`: no `fill` or `stroke` attribute in `sprite()` holds a literal colour; each is a
   `var(--ms-…, <fallback>)`; the `<style>` has a `prefers-color-scheme: dark` block.
5. `glyph.deterministic`: two calls of `sprite()` give equal bytes; no numeric attribute holds a
   `.`.
6. `glyph.enums-match-domain`: the serialized names of every variant of `OperatorFamily`,
   `PipelineStage` and `EndpointRole` equal the `enum` of
   `crates/modelspec-observatory/schema/types/modelspec.diagram.<Enum>.schema.json`, in order.
7. `task core-no-vulkan` passes, `cargo check -p modelspec-diagram --features ts` exits 0, and
   `git diff --exit-code crates/modelspec-core/Cargo.toml` is empty.

## Out of scope

- Drawing a graph (`story:ir-gallery-renderer`); any page (`story:ir-gallery-atoms`).
- Generating TypeScript from the enums (`story:observatory-architecture-graph` enables `ts`).
