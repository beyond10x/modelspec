---
format: aep.planning-md/3
id: story:decider-loss-terms-and-rank-probe
kind: story
status: draft
title: Per-term loss records and an effective_rank probe in the run record
summary: record = [...] and probe effective_rank(x) every N in the train block, as metric samples on CPU and Vulkan.
relations:
- decomposes: epic:decider-training-support
- serves: vision:modelspec
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/train_record_terms.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/training.rs
- confidence: cited
  path: crates/modelspec-core/src/lower.rs
- confidence: cited
  path: crates/modelspec-core/src/resolve.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/ast.rs
- confidence: cited
  path: crates/modelspec-core/src/syntax/parser.rs
- confidence: cited
  path: crates/modelspec-core/src/types.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-docs/src/grammar.rs
- confidence: cited
  path: crates/modelspec-observatory/schema
- confidence: cited
  path: crates/modelspec-runtime/schema
- confidence: inferred
  path: crates/modelspec-runtime/src/train/probe.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/src/vulkan/stepwise.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/effective_rank.rs
- confidence: cited
  path: spec/domains/observatory.yaml
- confidence: cited
  path: spec/domains/training.yaml
- confidence: cited
  path: ui/src/generated
- confidence: cited
  path: website
revision: 3
---
# Per-term loss records and an effective_rank probe in the run record

`depends_on: none`

## Outcome

A `train` block may name scalar `let`s of `forward` to record, `record = [l_jepa, l_sig,
l_dec];`, and a rank-2 `let` to probe, `probe effective_rank(s) every 100;`. Every
`StepCompleted` then carries one metric sample per recorded term, named as the `let`, with the
value that step's loss was computed from; and every step that is a multiple of the probe's
interval, and the run's first and last, carries `effective_rank:<let>`, the RankMe effective rank
(Garrido et al. 2023: `exp` of the entropy of the normalised singular values) of that value,
computed on the host in `f64`. Both work on `--backend cpu` and `--backend vulkan`, and both reach
`--record`, `--serve` and the observatory UI through the existing metric samples.

## Context

- Gap row 2. Decider design § 5.1: "The embedding rank proxy is measured every 100 steps either
  way", and the choice between SIGReg and the EMA fallback (l. 213–214) needs `l_jepa`, `l_sig`
  and `l_dec` apart. Today only the summed loss reaches the record (`decider.mspec`: "GAP(2): the
  three terms reach the run record only as their weighted sum").
- `MetricSample` and `StepCompleted.metrics` exist (`spec/domains/observatory.yaml`), filled today
  only by held-out evaluation (`heldout_mlm_loss`, `heldout_mlm_accuracy`;
  `crates/modelspec-cli/src/train.rs:844-943`).
- A Vulkan run keeps values on the device between host waits; per-term scalars travel with the
  loss slots, and a probed value is read back at the probe's steps only, one extra download (and,
  on the whole-run runner, one extra host wait) per probed step.

## Spec first

Before any code: `spec/domains/training.yaml` `TrainingRun` gains `recorded_terms`
(`List<String>`) and `probes` (`List<modelspec.training.Probe>`, a new struct: `kind`, an enum
`EffectiveRank`; `value`, the `let` name; `every`, an integer ≥ 1);
`spec/domains/observatory.yaml` states in `StepCompleted.metrics` which names the terms and the
probe produce and on which steps. Then `task spec-validate`, `task schema-generate` (both
`schema/` trees), `task ui-types`.

## Scope

- `spec/domains/training.yaml`, `spec/domains/observatory.yaml`,
  `crates/modelspec-observatory/schema/`, `crates/modelspec-runtime/schema/`, `ui/src/generated/`.
- `crates/modelspec-core/src/syntax/parser.rs`, `syntax/ast.rs`, `resolve.rs`, `types.rs`,
  `ir/training.rs` (`TrainConfig`), `lower.rs` (the named values kept live through DCE).
- `crates/modelspec-runtime/src/train/runner.rs`, `crates/modelspec-runtime/src/train/probe.rs`
  (new), `crates/modelspec-runtime/src/vulkan/runner.rs`,
  `crates/modelspec-runtime/src/vulkan/stepwise.rs`.
- `crates/modelspec-cli/src/train.rs`.
- Tests (new): `crates/modelspec-cli/tests/train_record_terms.rs`,
  `crates/modelspec-runtime/tests/effective_rank.rs`.
- `DESIGN.md` § 2 (the two settings), § 8; `crates/modelspec-docs/src/grammar.rs` and
  `diagnostics.rs`, then `task docs-generate` (`website/`). One line in `CHANGELOG.md`.

## Acceptance

- **effective-rank-values**: `cargo test -p modelspec-runtime --test effective_rank`: the probe
  gives `n` for an `n × n` identity, `1` for a rank-1 matrix, and the value the test computes from
  a known singular spectrum, each within `1e-9`.
- **record-terms-cpu**: `cargo test -p modelspec-cli --test train_record_terms`: a fixture whose
  loss is `add(a, mul(b, 0.5))` with `record = [a, b];` writes, in `--record`, a metric sample
  `a` and `b` at every step whose weighted sum equals the step's loss within `TOL_F32`.
- **probe-steps-cpu**: with `probe effective_rank(h) every 3;` over 7 steps, exactly steps 0, 3, 6
  and 7 carry `effective_rank:h`.
- **record-refusals**: a recorded name that is not a `let` of `forward`, a non-scalar recorded
  term, a probe of a value that is not rank-2, and `every 0` are each refused by `check` with
  their code.
- **record-terms-vulkan** (`#[ignore]`d, `--ignored`): the same fixtures on `--backend vulkan`,
  each term and probe within the DESIGN.md § 9 gate 5 tolerance of the CPU run, and `--stats`
  showing host waits increased by the probed steps only.
- `task ci` exits 0, including `task spec-validate`, `task schema-check`, `task ui-types-check`
  and `task docs-check`.

## Out of scope

- New UI views: the samples appear in the existing metric charts.
- Probes other than `effective_rank`; activation statistics (`story:observatory-activation-statistics`).
- The SIGReg-versus-EMA decision itself: the decider repository reads the series.
