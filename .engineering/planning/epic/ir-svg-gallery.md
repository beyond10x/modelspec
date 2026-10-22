---
format: aep.planning-md/3
id: epic:ir-svg-gallery
kind: epic
status: draft
title: 'IR diagrams as SVG: a component gallery in the documentation'
relations:
- serves: vision:modelspec
revision: 1
---
# IR diagrams as SVG: a component gallery in the documentation

## Outcome

The documentation site has its own section that shows the ModelSpec intermediate representation
as SVG diagrams, arranged like a UI component library: one entry per atomic element of the
computation graph (each operator of the registry in DESIGN.md), then composed layers, then whole
modules and the examples in `examples/`. Every diagram is generated from the IR by a Rust crate in
this repository; the site embeds the files and no SVG is written by hand.

The operator stated the idea on 2026-10-22, verbatim:

> "So file an idea for model spec. So for the documentation, I mean, we have this this tooling
> where we create the intermediate representation. Yeah, maybe we create some little view helper
> which would be able to render this in a really nice way, you know, all the different modules and
> such. So maybe there's a separate section just for that, like a component library, you know,
> like a UI component library. We do the same with the atomic elements we have, basically, to
> build our graph, our computation graph, up to later the layers and such. Yeah, so we want to be
> able to visualize the thing, ideally, maybe ideally, as an SVG, probably as an SVG. Yeah, do
> that, build it into the documentation. I mean, for the model spec agent, of course."

## Constraints

- The renderer is Rust, inside the Cargo workspace. No Node, Python or external layout binary
  (such as Graphviz) at generation time; `website/` only embeds the files.
- Input is the verified IR the compiler already produces (`modelspec ir --stage tensor|training`),
  not a second parse of `.mspec` source.
- Output is deterministic: the same IR yields the same bytes, so the files can be committed and
  drift-checked.
- The SVG files sit beside the other generated pages and are held by the drift check that already
  guards them (`task docs-check`, inside `task ci`); a stale or hand-edited diagram fails the gate.
- `modelspec-core` gains no rendering dependency beyond what DESIGN.md § 0 allows.

## Levels

| level | what one entry shows |
|---|---|
| atom | one operator of the registry: its inputs, output, shapes and dtype, plus its gradient rule in the training IR |
| layer | a composed block, such as a dense layer with its activation, as a subgraph |
| module | a whole model of `examples/`, forward graph, and the training step (forward, backward, update) |

## Out of scope

- Interactive diagrams (pan, zoom, hover) and any client-side JavaScript renderer.
- Live diagrams of a running training run; that is `epic:training-observatory`.
- Rendering the Kernel IR or the execution plan of M1; a later story may add it.
