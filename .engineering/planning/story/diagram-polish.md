---
format: aep.planning-md/3
id: story:diagram-polish
kind: story
status: draft
title: 'Diagram polish: source and diagram as tabs, pan and zoom, compact nodes, clean edges'
relations:
- decomposes: epic:ir-svg-gallery
- depends_on: story:diagram-docs-blocks
revision: 1
---
# Diagram polish: source and diagram as tabs, pan and zoom, compact nodes, clean edges

## Context

Operator review of the local site build, 2026-10-29, low priority. The diagrams sit under each
` ```modelspec ` block (`story:diagram-docs-blocks`). Asked for:

- a tabbed view: one component in this repository, used by the Docusaurus pages, that takes the
  modelspec source and shows two tabs, "Source" and "Diagram", the way documentation sites switch
  between programming languages;
- edges and arrows that do not overlap;
- large models (BERT-tiny) that are usable: the boxes are too big, the canvas cannot be zoomed or
  moved, and the diagram is cut off.

## What it adds

- `website/src/components/ModelSpecProgram.tsx` (or the existing `ModelDiagram.tsx` extended): the
  modelspec source as a code block in tab 1 and its diagram in tab 2, using Docusaurus `Tabs` with a
  `groupId` so the choice holds across the page and the site; `modelspec-docs` emits this component
  for every whole block instead of a code block followed by a figure.
- A pan and zoom canvas for the diagram tab: wheel or pinch zoom, drag to move, a "fit" button, and
  the initial view fitted to the tab's width, keyboard operable.
- `crates/modelspec-diagram`: a compact node style for large graphs (smaller boxes, shape on one
  line), and edge routing that keeps edges apart (no two edges drawn over each other, arrows ending on
  distinct points of a node's border).
- A visual review of the built site at 1440 and 390, light and dark, on two-layer-relu, BERT-tiny and
  the layers gallery, with screenshots.

## Acceptance

1. Every whole modelspec block on the site renders as a Source/Diagram tab pair; switching on one page
   switches the others.
2. BERT-tiny's diagram opens fitted to the column, zooms to node labels at body text size, and moves by
   drag and by keyboard; nothing is cut off.
3. A renderer test: on BERT-tiny's model view, no two edge paths share a segment and no two arrowheads
   end on the same point.
4. Screenshots of the review are recorded in the PR.
