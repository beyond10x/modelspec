---
format: aep.planning-md/3
id: story:m0-cpu-sgd-training
kind: story
status: implemented
title: SGD training step and modelspec train --backend cpu on linear regression
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
- depends_on: story:m0-reverse-mode-autodiff
- depends_on: story:m0-lower-tensor-ir
- depends_on: story:m0-cpu-interpreter
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-cli/src/ir.rs
- confidence: cited
  path: crates/modelspec-cli/src/main.rs
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: cited
  path: crates/modelspec-cli/tests/exit_codes.rs
- confidence: cited
  path: crates/modelspec-cli/tests/golden/linear_regression.training.ir
- confidence: cited
  path: crates/modelspec-cli/tests/ir_tensor.rs
- confidence: cited
  path: crates/modelspec-cli/tests/train_linear.rs
- confidence: cited
  path: crates/modelspec-cli/tests/train_linear_adversary.rs
- confidence: cited
  path: crates/modelspec-core/src/compile.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/print.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/training.rs
- confidence: cited
  path: crates/modelspec-core/src/ir/verify.rs
- confidence: cited
  path: crates/modelspec-core/tests/training_verify.rs
- confidence: cited
  path: crates/modelspec-core/tests/training_verify_adversary.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/init.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/optimizer.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: cited
  path: crates/modelspec-runtime/tests/train_adversary.rs
revision: 31
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 20, decided_on: {"recorded":{"review_outcome":3}}}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:27Z", actor: "human:timo", revision: 21, decided_on: {"recorded":{"review_outcome":3}}}
- {from: "active", to: "implemented", at: "2026-10-21T14:53:41Z", actor: "human:timo", revision: 31, decided_on: {"recorded":{"test_result":1,"review_outcome":5,"verification":1}}}
---
# SGD training step and `modelspec train --backend cpu` on linear regression

## Context

Closes the M0 vertical slice (DESIGN.md § 11 iteration 1: "Train linear regression"): the whole
training step (forward, loss, backward, SGD) is one compiled program run by the CPU interpreter
(DESIGN.md § 0 invariant 5; ADR-005). Implements DESIGN.md § 3.1 Training IR (`Optimizer`
subgraph; reads and writes of parameter state carry ordered effect tokens; an update yields a new
state version), § 3.3 pass 5, § 2.3 reproducible initialization, § 6 training host (seed and state
initialization, step counter, loss telemetry), § 8 verbs `train --backend cpu` and
`ir --stage training`, and DESIGN.md § 9 gate 4. Owns the third epic acceptance bullet and the
effect part of the epic's "type/shape/effect checks": the legal-effects verifier check of
DESIGN.md § 3.2.

## Lands in

- `crates/modelspec-core/src/ir/training.rs`: `Optimizer` subgraph for SGD,
  `p_next = p - lr * grad` per parameter, added to `training.program` as one output per parameter.
  Effect tokens and state versions are defined here as Training IR metadata, not as `TensorOp`s (so
  `ir/tensor.rs` is not edited): each parameter carries a state version, each `ParamRead` and each
  `p_next` output carries an ordered effect token, and an update yields version `n + 1` from `n`
  (DESIGN.md § 3.1, § 4). Also `TrainConfig { steps, lr, seed, input, target }`, built from the HIR
  `train` block, which carries the file's SGD configuration to the runtime.
- `crates/modelspec-core/src/ir/verify.rs`: the legal-effects check of DESIGN.md § 3.2, extending
  the Training IR checks of story `m0-reverse-mode-autodiff`: per parameter at most one write per
  step, no write token ordered before the last read of the old version, and every write producing
  exactly the next version. Its failures are verifier errors.
- `crates/modelspec-core/src/compile.rs`: `compile_training(path, source) -> CompiledTraining
  { training: TrainingProgram, forward: LoweredProgram, config: TrainConfig }`: `compile_hir`, then
  `TrainConfig` built from its `HirTrain`; lower and DCE into `forward` (the pre-autodiff program
  with its `prediction` value); then autodiff, SGD attachment and verify (including the
  legal-effects check) into `training`.
- `crates/modelspec-core/src/ir/print.rs`: Training IR printing with subgraph tags and state versions.
- `crates/modelspec-runtime/src/train/init.rs`: counter-based PRNG Philox4x32-10 keyed by the global
  seed, with the counter carrying a stream tag, the canonical parameter id (the first 8 bytes of
  `blake3` of the canonical path, little-endian) and the element index, so values never depend on
  compiler iteration order (DESIGN.md § 2.3). `xavier_uniform` on `[In, Out]` draws from
  `U(-a, a)`, `a = sqrt(6 / (In + Out))`. DESIGN.md § 2.3 names Philox4x32-10; record this key,
  counter and stream derivation there and bump the revision line.
- `crates/modelspec-runtime/src/train/data.rs`: M0 synthetic teacher data. Inputs are standard
  normal (Box-Muller over the PRNG, data stream); targets are the model `forward` under teacher
  parameters drawn with `xavier_uniform` on the teacher stream, computed by running
  `forward.program` on the interpreter with a zero `target` input and reading the intermediate value
  `forward.prediction` (the interpreter returns intermediates on request). The
  teacher parameters are the known solution of DESIGN.md § 9 gate 4. Refused with a diagnostic when
  the `train` block does not have exactly one `input` and one `target` of the model output type.
  Margin: for `[8, 4]` standard normal inputs the expected smallest eigenvalue of the loss Hessian
  (about 0.17) leaves a parameter error near 2e-4 after 500 steps at `lr = 0.1`; if seed 42 still
  misses the thresholds below, report it rather than loosening them.
- `crates/modelspec-runtime/src/train/{optimizer,runner}.rs`: SGD configuration from `TrainConfig`;
  the step loop runs `&training.program` on the CPU interpreter unchanged, then applies each `p_next`
  output as the parameter's next version in effect-token order and records the loss per step.
  `src/cpu/interpreter.rs` is not edited.
- `crates/modelspec-cli/src/train.rs`: `modelspec train <file> --backend cpu` prints the loss at
  step 0, every 50 steps and at the end, then each parameter by canonical path; `--backend vulkan`
  exits 1 with a diagnostic that the backend is not available in M0.
- `crates/modelspec-cli/src/ir.rs`: `--stage training`, with golden
  `crates/modelspec-cli/tests/golden/linear_regression.training.ir`.
- Tests: `crates/modelspec-cli/tests/train_linear.rs` (the acceptance), plus: two runs with seed 42
  print identical bytes; seeds 42 and 43 print different initial parameters; the training golden;
  `crates/modelspec-core/tests/training_verify.rs`, where a builder-made training program with a
  write ordered before the last read of the old version is refused by the legal-effects check.

## Out of scope

- A data file format, `--data` and `.msdata` (DESIGN.md § 12 "Data interface" is open); M0 data is
  synthetic only.
- Checkpoints and `--checkpoint` (M1, `spec/domains/checkpoint.yaml`); inference; `--steps` override.
- Writing a training-run record (see Domain relations).
- Optimizers other than SGD; training `examples/tiny_mlp.mspec` (modules, generics, Vulkan device).

## Domain relations

This story relies on none. Two typed relations bear on it and are deliberately not used, both in
`spec/domains/training.yaml`, entity `modelspec.training.TrainingRun`:

- relation `artifact`: references `modelspec.artifact.CompiledArtifact`, cardinality one, via
  `artifact_id`;
- relation `device`: references `modelspec.device.DeviceProfile`, cardinality one, via the optional
  `device_profile_id`, absent for a CPU run.

M0 produces no `CompiledArtifact`, so it writes no `TrainingRun`. The record, with its
`OptimizerConfig` and `RngState` (`RngAlgorithm::Philox4x32x10`), is M1 work.

## Acceptance

`cargo test -p modelspec-cli --test train_linear` passes: it runs
`modelspec train examples/linear_regression.mspec --backend cpu` (seed 42 and 500 steps, as in the
file) and asserts exit 0, a final loss at most 1e-3 times the step-0 loss, and every element of the
learned `LinearReg.w` within 1e-2 of the teacher weights.
