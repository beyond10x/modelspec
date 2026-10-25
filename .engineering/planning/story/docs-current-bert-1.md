---
format: aep.planning-md/3
id: story:docs-current-bert-1
kind: story
status: active
title: The documentation site, README and AGENTS.md brought current with the BERT operator chain, and the site review's own-file defects fixed
relations:
- serves: vision:modelspec
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: README.md
- confidence: cited
  path: crates/modelspec-docs/src/**
- confidence: cited
  path: docs/pitch/index.html
- confidence: cited
  path: website/data/**
- confidence: cited
  path: website/docs/**
- confidence: cited
  path: website/product.json
- confidence: cited
  path: website/sidebars.ts
- confidence: cited
  path: website/static/img/benchmarks/**
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T09:50:24Z", actor: "human:timo", revision: 3}
- {from: "proposed", to: "active", at: "2026-10-25T09:50:24Z", actor: "human:timo", revision: 4}
---
# The documentation site, README and AGENTS.md brought current with the BERT operator chain, and the site review's own-file defects fixed

## Context

The operator asked on 2026-10-25 to "make sure we keep the docs up-to-date". The hand-written
pages under `website/docs/` were last brought current for 0.7.1 (`website/docs/status.mdx` line
10 says "0.7.1 is the latest release"; `crates/modelspec-docs/src/status.rs` `AS_OF` is
2026-10-23 and its `CAPABILITIES` stop before 0.8.0). Since then 0.8.0, 0.9.0 and 0.10.0 shipped
seven operators (`embedding`, `add_rows`, `gelu`, `layer_norm`, `softmax`, `attention`,
`cross_entropy`), `i32` tensors for ids and labels, `modelspec data pack` and `modelspec
tokenizer`, and the Unreleased section of `CHANGELOG.md` holds `dropout` and `train --data` on a
packed masked-language-model directory. A docs-site review on 2026-10-25 (5 screeners, 37
findings, 5 high and 13 medium confirmed) found the content drift above and a set of layout and
theme defects, some in this repository's own files and some in the shared docs template.

## What this story decides

- Every hand-written page under `website/docs/` (`index.md`, `getting-started.md`,
  `concepts/language.md`, `concepts/pipeline.md`, `concepts/training.md`, `guides/*.md`,
  `roadmap.md`, `status.mdx`) states what `main` does today, checked against the code, the tests,
  `modelspec --help` and `CHANGELOG.md` since 0.7.1. Unshipped things are labelled with a status
  admonition. The CHANGELOG review since 0.7.1 is recorded: each consumer-relevant entry names the
  page that shows it, or a one-line reason why no page changes.
- New concept and guide pages where a shipped capability has none: the BERT operators (what
  `embedding`, `add_rows`, `gelu`, `layer_norm`, `softmax`, `attention`, `cross_entropy` and
  `dropout` compute, their `i32` operands, their gradients, which have a Vulkan kernel) and the
  data path (`data pack`, `tokenizer`, `train --data` on a packed directory, the `data` line). One
  idea per page; every command on a page run in the tree and its output pasted.
- `crates/modelspec-docs/src/status.rs`: `CAPABILITIES` gains a row per shipped capability of
  0.8.0 to 0.10.0 and of Unreleased, each `shipped` row naming the test or task that holds it;
  `AS_OF` is the date of the check. `website/data/status.json` and `status.mdx` follow.
- The generated pages stay generated: a defect in a generated page is fixed in its generator
  (`cli.rs`: the `--help` usage lines reflowed so no line runs past the code box, the
  `http://ADDR/` placeholder written so Docusaurus does not auto-link it; `benchmarks.rs`: the
  SVG charts readable in both themes and at 390 px wide, with no hard-coded white background and
  labels no smaller than 12 px at the rendered size).
- The pitch deck (`docs/pitch/index.html`) follows `prefers-color-scheme` and the site's theme
  rather than a fixed dark background.
- `README.md` and `AGENTS.md` name the same commands, crates and pages as the site; AGENTS.md's
  invariant tests still exist.
- The site review's fix list is answered per id in a file under the round directory the
  coordinator names: `fixed` with the commit and the file, or `template` with the docs-system
  file it lives in (routed to the template's owner, not fixed here), or `not a defect` with the
  reason.

## Domain relations

No new noun. The status page reads the planning store and the tests; the data path pages
describe `modelspec.data.PackedDataset`, `TrainingDataBinding` and `Tokenizer` of
`spec/domains/data.yaml` as they are.

## Scope

- `website/docs/**` hand-written pages (new and changed), `website/product.json`, `website/sidebars.ts`
  if a new page needs a sidebar entry.
- `crates/modelspec-docs/src/{status.rs,cli.rs,benchmarks.rs,diagnostics.rs}` and the generated
  output they own (`website/docs/reference/**`, `website/docs/examples/**`, `website/data/**`,
  `website/docs/benchmarks.md`, `website/static/img/benchmarks/**`) through `task docs-generate`.
- `crates/modelspec-cli/src/**` only for a `--help` text a reference-page defect comes from.
- `docs/pitch/index.html`.
- `README.md`, `AGENTS.md`, `CHANGELOG.md` (one line under Unreleased / Changed).

## Acceptance

1. `task docs-check`, `task check` and the site build (`task website`) exit 0 on the tree.
2. `curl -s https://beyond10x.github.io/modelspec/docs/status/` after Stage B, and the tree's
   built `website/build/docs/status/index.html` before it, contain "0.10.0" and not
   "0.7.1 is the latest release"; `/docs/concepts/language/` does not contain "nine operators"
   or "only element type"; `/docs/concepts/pipeline/` and `/docs/roadmap/` carry no "Not built"
   beside Kernel IR, Vulkan execution or an operator that shipped.
3. `website/data/status.json` has `asOf` 2026-10-25 or later and a `shipped` row for every
   operator of 0.8.0 to 0.10.0, for `data pack`, `tokenizer`, `cross_entropy` and `dropout`.
4. Every `shipped` row's evidence exists: `modelspec-docs generate` exits 0 (it refuses a missing
   test or task).
5. The getting-started transcript equals the output of the command it shows, run in the tree.
6. The fix-list answer file names every one of the 37 ids with `fixed`, `template` or
   `not a defect`; every `fixed` id names a commit in the branch and a file; every `template` id
   names a docs-system file.
7. The benchmark chart SVGs declare no `fill="#fff"`-style background and their label font size
   renders at 12 px or more at 390 px viewport width (checked with a Playwright shot at 390x844
   in both themes, the shots kept beside the fix-list answer file).
8. `grep -q "docs-generate" AGENTS.md` and every `path::function` AGENTS.md names exists.

## Out of scope

- Template defects (wide tables cut at 1280, navbar brand at 1024, inline code breaking at a
  hyphen, sidebar under the footer, link contrast, 404 head fields, labels under 12 px in
  template components): listed for the docs-system owner, not fixed here.
- Stage C (the repository already serves its own site at `/modelspec/`).
- The IR gallery (`epic:ir-svg-gallery`).
