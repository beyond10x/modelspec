---
format: aep.planning-md/3
id: story:diagram-cli
kind: story
status: draft
title: 'modelspec diagram: any level of any view as SVG or JSON'
summary: modelspec diagram FILE with --view, --scope, --depth, --expand, --format svg|json|graph and --out DIR, diagnostics and exit codes as modelspec ir.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:diagram-pipelines
- depends_on: story:ir-gallery-atoms
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/Cargo.toml
- confidence: inferred
  path: crates/modelspec-cli/src/diagram.rs
- confidence: cited
  path: crates/modelspec-cli/src/lib.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/diagram_cli.rs
- confidence: cited
  path: website/docs/reference/cli.md
revision: 5
---
# `modelspec diagram`: any level of any view as SVG or JSON

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.7. The docs and the observatory draw through
`modelspec-diagram`; a person or an agent draws from the command line. The command compiles a file
as `modelspec ir` does (`crates/modelspec-cli/src/ir.rs`; subcommands in
`crates/modelspec-cli/src/lib.rs:41`) and prints one level, or writes every level.

## What it adds

- `crates/modelspec-cli/src/diagram.rs` and `Command::Diagram` in `lib.rs`:
  `modelspec diagram <file> [--view model|train|eval|infer] [--scope <key>] [--depth <n>]
  [--expand <key>]... [--format svg|json|graph] [--out <dir>]`. It prints the SVG, the
  `DiagramRender` JSON or the `DiagramGraph` JSON to stdout. With `--out` it writes every scope's
  level as `<dir>/<scope-key>.svg`, composite links pointing at sibling files.
- Diagnostics and exit codes as `modelspec ir`; an unknown `--scope` or `--expand` key exits 2 and
  lists the graph's keys.
- The CLI reference page is regenerated (`crates/modelspec-docs/src/cli.rs`, `task docs-generate`).

## Scope

- `crates/modelspec-cli/src/diagram.rs` (new), `src/lib.rs` (one variant; cited),
  `Cargo.toml` (add `modelspec-diagram`; cited), `tests/diagram_cli.rs` (new).
- `Cargo.lock`: the new dependency rewrites it, as `story:ir-gallery-atoms` does for
  `modelspec-docs`; this story depends on that one, so the two lock changes land in order.
- `website/docs/reference/cli.md` (generated); `CHANGELOG.md` one line; DESIGN.md § 8 one line.

## Acceptance

Tests in `crates/modelspec-cli/tests/diagram_cli.rs` (`cargo test -p modelspec-cli`, `task ci`):

1. `diagram_cli.svg-default`: `modelspec diagram crates/modelspec-docs/models/two_layer_relu.mspec`
   exits 0 and prints the same bytes as `modelspec_diagram` renders for the model view at depth 1.
2. `diagram_cli.views-and-formats`: for `examples/linear_regression.mspec`, each of the four views
   in each of the three formats exits 0. The `json` and `graph` outputs validate against the
   `DiagramRender` and `DiagramGraph` JSON Schemas.
3. `diagram_cli.out-dir`: `--view train --out <tmp>` writes one file per scope of the graph; every
   composite `href` names a file in `<tmp>`.
4. `diagram_cli.refusals`: a file with a shape error exits 1 with the diagnostic `modelspec ir`
   prints for it; `--scope nope` exits 2 listing the keys.
5. `task docs-check` exits 0 with the regenerated CLI reference.

## Out of scope

- `--function` (`story:diagram-infer-function-view`).
