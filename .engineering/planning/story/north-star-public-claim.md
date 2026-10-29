---
format: aep.planning-md/3
id: story:north-star-public-claim
kind: story
status: implemented
title: 'ModelSpec''s north star is its public claim: faster training through a runtime compiled for one target'
relations:
- serves: vision:modelspec
scope:
- confidence: cited
  path: .engineering/planning/vision/modelspec.md
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: README.md
- confidence: cited
  path: docs/pitch/index.html
- confidence: cited
  path: website/docs/concepts
- confidence: cited
  path: website/docs/index.md
- confidence: cited
  path: website/docs/roadmap.md
- confidence: cited
  path: website/product.json
- confidence: cited
  path: website/sidebars.ts
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-28T11:30:33Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-10-28T11:30:34Z", actor: "human:timo", revision: 4}
- {from: "active", to: "implemented", at: "2026-10-29T08:26:20Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# ModelSpec's north star is its public claim

## Context

`vision:modelspec`. People shown ModelSpec did not understand what it is for: the site and README
said what it does (a language and compiler; no Python, PyTorch or CUDA) but not why. The operator
stated the intent on 2026-10-28: because the whole training step is one declared, typed program,
the compiler can compile it into a runtime built for one use case, one operating system and one GPU
chipset, and so train much faster than a general-purpose framework; because every kernel, fusion
and pass is checked against the CPU reference on the same IR, aggressive optimizations of the
compiler itself are safe to try. BERT-tiny against PyTorch on the RTX 3090 is the worked example:
the baseline (released 0.21.0 block figures, the full-model figures with 0.22.0), then A (the block
within 3× of PyTorch eager), E (full BERT-tiny at parity with PyTorch eager) and F (faster than
`torch.compile` per step and to the held-out target).

The speed is the goal, not shipped: every page that states it labels it planned and gives today's
released figures. One target exists (Vulkan compute and SPIR-V on one RTX 3090 on Linux); the
DESIGN.md § 0 non-goals stay.

## Scope

- `.engineering/planning/vision/modelspec.md` (the `## North star` section, through `aep plan
  artifact body`).
- `website/product.json`, `website/docs/index.md`, `website/docs/concepts/north-star.md` (new),
  `website/docs/roadmap.md`, `website/sidebars.ts`, and the `sidebar_position` lines of the other
  `website/docs/concepts/*.md` pages.
- `README.md`, `docs/pitch/index.html`, `AGENTS.md` § Serves.
- Not: `website/docs/benchmarks.md`, `website/docs/runs/`, `benchmarks/`, `CHANGELOG.md`,
  `opponents/`.

## Acceptance

- `aep plan artifact show vision:modelspec` prints a `## North star` section before `## Objective`
  stating the claim and the BERT-tiny ladder, and its Direction section names the safe
  self-improvement of the compiler as the practical form of the north star.
- The first screen of the landing page (`website/product.json` eyebrow, promise, lede and first
  features section) and the first two paragraphs of `website/docs/index.md` say what ModelSpec is
  for (faster training through a runtime compiled for one target), why the approach makes that
  possible, and where it is today, with the word "planned" on the speed claim.
- `website/docs/concepts/north-star.md` exists, is listed in `website/sidebars.ts`, carries title,
  sidebar_position, description, lede and source, quotes the batch-8 transformer-block row of
  `website/docs/benchmarks.md` exactly, and marks A, E and F as planned.
- `website/docs/roadmap.md` opens with one north-star paragraph linking that page and lists A, E
  and F as the near-term path, with no planning ids in prose.
- The first paragraph of `README.md` says what ModelSpec is for before what it is; the status line
  is unchanged.
- Slide 1 of `docs/pitch/index.html` carries the claim, and the deck still has 18 slides.
- `task docs-check`, `task ci` (the repository's gate) and the site build (`task website`) exit 0.
