---
format: aep.planning-md/3
id: story:run-record-names-numerics-class
kind: story
status: draft
title: The --record event stream names the run's numerics class and fast_math
relations:
- serves: vision:modelspec
- depends_on: story:fast-math-mode
revision: 1
---
# The `--record` event stream names the run's numerics class and `fast_math`

## Why

Adversary pass 1 on story:fast-math-mode (review-result:adversary-fastmath-pass-1, finding
ADV-FM-2): ADR-008 (DESIGN.md § 12) says "The run record and the `numerics class` line of
`--stats` name the class". The runner's report line (`crates/modelspec-runtime/src/vulkan/runner.rs:128`)
names it; the `train --record` event stream (`crates/modelspec-observatory/src/event.rs:128`,
`RunStarted`) carries only `steps` and `start_step`, so a recorded run cannot be told from a
deterministic one after the fact. Pre-existing: the base text said the same and
story:fast-math-mode did not touch the observatory.

Outcome: `RunStarted` carries `numerics_class` (`bit_for_bit` | `tolerance`) and `fast_math`
(bool); the observatory UI shows both on the run header; the shared types are regenerated from
Rust and the drift check passes.

## Acceptance

- **run-started-names-class**: `train --backend vulkan --record <file>` writes a `RunStarted`
  event whose `numerics_class` is the class the `--stats` line prints and whose `fast_math` is
  the artifact's; a test runs the MNIST example with and without `--fast-math` and reads both
  back (the ignored case `the_run_record_of_a_fast_math_run_names_the_class_tolerance` in
  `crates/modelspec-cli/tests/adversary_fastmath_device.rs` is un-ignored and passes).
- **cpu-run-names-class**: `train --backend cpu --record` writes `numerics_class: bit_for_bit`
  and `fast_math: false`.
- **schema-regenerated**: the event schema and the UI's generated types carry the two fields and
  `task docs-check` and the UI type drift check pass; an older stream without the fields still
  loads (the fields are optional on read).
- **ui-shows-class**: the observatory run header shows the class and the flag.
- `task ci` exits 0.

## Scope

- `crates/modelspec-observatory/src/event.rs`
- `crates/modelspec-cli/src/train.rs`
- `crates/modelspec-cli/tests/adversary_fastmath_device.rs`
- `ui/src/`
