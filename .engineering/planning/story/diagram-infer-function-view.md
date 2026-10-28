---
format: aep.planning-md/3
id: story:diagram-infer-function-view
kind: story
status: draft
title: 'Infer view of one model function: the diagram of infer --data --function'
summary: The infer view drawn for one model function, from the program infer --data --function lowers, with modelspec diagram --view infer --function.
relations:
- decomposes: epic:ir-svg-gallery
- serves: vision:modelspec
- depends_on: story:diagram-pipelines
- depends_on: story:diagram-cli
- depends_on: story:decider-infer-on-data
scope:
- confidence: inferred
  path: crates/modelspec-cli/src/diagram.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/diagram_cli.rs
- confidence: inferred
  path: crates/modelspec-diagram/src/pipeline.rs
- confidence: inferred
  path: crates/modelspec-diagram/tests/infer_function.rs
revision: 2
---
# Infer view of one model function: the diagram of `infer --data --function`

## Context

`epic:ir-svg-gallery`, DESIGN.md § 13.4. `story:decider-infer-on-data` (draft) adds
`infer --data DIR --function NAME --out FILE`, which lowers one model function as its own program
with its arguments as inputs. Once that lowering exists, the `infer` view can draw it: the named
function's scope as root, its arguments as Data, its result as the PredictionSink.

## What it adds

- `crates/modelspec-diagram/src/pipeline.rs`: an `infer` graph for `function: Some(name)`, built
  from the program the infer-on-data lowering produces; `graph_id` gains `:<name>` and `ir_hash` is
  blake3 of that program's print (`spec/domains/diagram.yaml`, `DiagramGraph.ir_hash`).
- `modelspec diagram --view infer --function <name>` (`crates/modelspec-cli/src/diagram.rs`).

## Scope

- `crates/modelspec-diagram/src/pipeline.rs`, `tests/infer_function.rs` (new test).
- `crates/modelspec-cli/src/diagram.rs`, `crates/modelspec-cli/tests/diagram_cli.rs` (cited from
  `story:diagram-cli`).

## Acceptance

1. `pipeline.infer-function`: for the fixture `story:decider-infer-on-data` adds, the `infer` graph
   with `--function` of its named function has exactly the Operator nodes that function's lowered
   program has, its arguments under Data and its result into the PredictionSink endpoint.
2. `diagram_cli.infer-function-unknown`: `modelspec diagram <file> --view infer --function nope`
   exits 2 naming the model's functions; `--function` with any other view exits 2.
3. `task ci` exits 0.

## Out of scope

- The infer command itself (`story:decider-infer-on-data`).
