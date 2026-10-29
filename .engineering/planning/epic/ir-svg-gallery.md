---
format: aep.planning-md/3
id: epic:ir-svg-gallery
kind: epic
status: draft
title: 'Architecture diagrams: operator glyphs, any model, pipelines, zoom'
summary: A glyph per operator, any model drawn from its verified IR, training/eval/inference pipelines, zoom through composites; beside every modelspec program in the docs, in a gallery, on the CLI and in the observatory UI.
relations:
- serves: vision:modelspec
revision: 7
---
# Architecture diagrams: operator glyphs, any model, pipelines, zoom

## Outcome

Every program ModelSpec verifies can be drawn. Each operator of the registry, and each operation
the training step adds (gradients, the optimizer update, the dropout mask, the loss), has a glyph;
any model is drawn from its verified IR; training, evaluation and inference are drawn as pipelines
whose stages zoom into the model; and a composite (a stage, an inlined function, later a module or
layer instance) zooms in and out. Wherever the repository's documentation shows a ` ```modelspec `
program (`website/docs`, `README.md` and `DESIGN.md`), its diagram stands beside it; a block the
compiler refuses on purpose is listed with the code it answers instead, and gets its diagram once it
compiles. The documentation site also has its own gallery, laid out like a UI component library:
atoms (the glyphs), layers, modules and pipelines. The observatory UI draws the same graph and zooms
in place.

The operator stated the gallery idea on 2026-10-22, verbatim:

> "So file an idea for model spec. So for the documentation, I mean, we have this this tooling
> where we create the intermediate representation. Yeah, maybe we create some little view helper
> which would be able to render this in a really nice way, you know, all the different modules and
> such. So maybe there's a separate section just for that, like a component library, you know,
> like a UI component library. We do the same with the atomic elements we have, basically, to
> build our graph, our computation graph, up to later the layers and such. Yeah, so we want to be
> able to visualize the thing, ideally, maybe ideally, as an SVG, probably as an SVG. Yeah, do
> that, build it into the documentation. I mean, for the model spec agent, of course."

and widened it on 2026-10-28, verbatim:

> "So what I also want to see now basically is in model spec we have all the operators, and
> technically each operator should get a symbol, a graphic we can paint basically. So technically it
> should be possible to render every model architecture. So this should also help us in visualizing
> pipelines from training to inference, and yeah, it would just be a really great visual thing to
> have. So that means also anywhere in the repo where in the docs we render code of model spec
> notation, we can also render the visual pendant of it. So make a design for that and yeah, do it.
> Also take into consideration that we might build up bigger modules or compositions out of smaller
> ones. So technically we could even, you know, like zoom in, zoom out. That would be awesome."

## Design

DESIGN.md § 13 (revision 0.81) holds every decision: glyphs (§ 13.1), the graph model and scope
provenance (§ 13.2), zoom (§ 13.3), pipelines (§ 13.4), layout (§ 13.5), documentation (§ 13.6),
the command line (§ 13.7) and where the code lives (§ 13.8).

## Specification

The nouns are `spec/domains/diagram.yaml` (`modelspec.diagram`): `DiagramGraph` (the whole graph of
one program in one view, every node with its scope) and `DiagramRender` (one laid-out projection,
`references` its graph), with `Scope`, `ScopeSegment`, `ScopeKey`, `Port`, `DiagramNode`,
`DiagramEdge`, `GlyphId`, `OperatorFamily`, `PipelineStage` (six stages; a checkpoint and a
prediction are `EndpointRole`s, never stages), `EndpointRole`, `NodeBox` and `EdgeRoute`. The glyph
enums are owned in Rust by `story:diagram-glyph-set`, the graph and render types by
`story:ir-gallery-renderer`. It replaces `modelspec.observatory.ModelGraph`, `GraphNode`,
`GraphEdge`, `IrStage` and `ModelGraphId`, which no code used, and settles their open question on
`artifact_id`: a graph names no artifact and carries the `ir_hash` a compiled artifact of the same
source records. `ess specify validate --path .` in `spec/`: `modelspec v1 — 9 file(s), valid`.

## Constraints

- All code that draws is Rust in the new crate `crates/modelspec-diagram` (DESIGN.md § 13.8). No
  Node, Python or external layout binary (such as Graphviz) at generation time, and no JavaScript
  layout. `website/` embeds the files through one small site component; the observatory UI is Vue 3
  and TypeScript and only pans, zooms and moves boxes the server laid out.
- The input is the verified IR the compiler already produces (`modelspec ir --stage
  tensor|training`), never a second parse of `.mspec` source. The scope of each node is recorded
  by lowering, autodiff and the optimizer attach in the IR's scope table, not inferred afterwards.
- The output is deterministic: the same IR and the same request give the same bytes, so the files
  are committed and drift-checked by `task docs-check` inside `task ci`.
- `modelspec-core` gains the scope table and no rendering dependency; `modelspec ir` output, the
  golden files and every `ir_hash` stay the same bytes.

## Levels

| level | what one entry shows |
|---|---|
| atom | one glyph: its operator, family and gradient glyph |
| layer | a composed block as a function scope, zoomable |
| module | a whole program of `examples/`, model view and pipeline views |
| pipeline | train, eval and infer, each stage a composite that zooms into the model |

## Landing order

| # | story | depends on |
|---|---|---|
| 1 | `story:diagram-glyph-set` | |
| 1 | `story:diagram-scope-provenance` | (parallel with 1 to 3) |
| 2 | `story:ir-gallery-renderer` | glyph set |
| 3 | `story:ir-gallery-atoms` (first visible result: the glyph gallery and the two-layer-relu diagram on its example page) | renderer |
| 4 | `story:diagram-zoom` | renderer, scope provenance |
| 5 | `story:diagram-pipelines` | zoom |
| 6 | `story:diagram-cli`, `story:diagram-docs-blocks`, `story:ir-gallery-layers` | pipelines and atoms (layers: zoom and atoms) |
| 7 | `story:ir-gallery-modules` | docs blocks, pipelines |
| 8 | `story:observatory-architecture-graph` (epic:training-observatory) | pipelines, cli |
| later | `story:diagram-infer-function-view` | pipelines, cli, `story:decider-infer-on-data` |

`Cargo.lock` changes in glyph set, renderer, atoms, cli and the observatory story, in that order,
each ordered after the one before by `depends_on`.

## Out of scope

- Rendering the Kernel IR or the execution plan of M1; a later story may add it.
- Live overlays of a running training run (statistics on nodes); that is `epic:training-observatory`.
- The architecture view of a replayed run (`story:observatory-architecture-graph`, Live only).
- Module and layer instances as scopes before DESIGN.md § 11 iteration 3; their `ScopeKind`s and
  glyphs exist now so the change that adds them only has to stamp them.
