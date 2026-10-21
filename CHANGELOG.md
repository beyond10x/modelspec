# Changelog

All notable changes to ModelSpec are recorded here, in the format of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). ModelSpec releases from source at
bare-version tags such as `0.1.0`; nothing is on a registry. Each release on GitHub carries the
`modelspec` CLI binary for Linux x86_64 and its SHA-256 checksum. Changes collect under
`[Unreleased]` until the next release; the detail of each change is in its story and commit
history.

## [Unreleased]

### Changed

- CPU reference: `matmul` runs in loop order `i`, `k`, `j` with the same bits as 0.1.1, and the `mnist_shape` benchmark (`cargo bench -p modelspec-runtime --bench mnist_shape`) prints images/s for the 784 → 64 → 10 shape (`cpu-matmul-row-order`)

## [0.1.1] - 2026-10-21

Documentation release: the website review's fixes and highlighted `modelspec` code blocks. The
compiler and runtime behave as in 0.1.0.

### Added

- Documentation site: `modelspec` code blocks are syntax-highlighted; the word lists are `website/data/modelspec.grammar.json`, generated from `syntax::lexer::KEYWORDS`, `resolve::OPERATORS` and the parser

### Changed

- Diagnostics reference: every code has its own anchor, such as `reference/diagnostics#e-type-012`
- CLI reference: an argument table without optional arguments drops the Required and Default columns, and each possible value of `--stage` and `--backend` has its own line
- Operators reference: the message of a refused call wraps between its parts
- `website/data/status.json` and the Status page: each shipped capability names the test or task that holds it
- Getting started: the install block uses `install -D` and says `~/.local/bin` must be on `PATH`; the binary links `libc`, `libm` and `libgcc_s`
- Roadmap: planned milestones carry the `planned` badge, and the exclusions list all of DESIGN.md § 1 and § 2.4
- Write a model: step 2 says where `forward` goes, and step 4 shows the `E-TYPE-012` output
- The language: `infer` is refused with `E-SYNTAX-002`; a top-level `target` is later, `target y:` in `train` is M0
- Pitch deck `docs/pitch/`: slides 13 and 15 show the 0.1.0 release, slide 8's loop arrows meet their boxes, tap, swipe and Escape work, and a portrait phone gets a notice

## [0.1.0] - 2026-10-21

The M0 semantic core (`epic:m0-semantic-core`, DESIGN.md revision 0.6). There is no GPU or Vulkan
backend yet: Vulkan execution is milestone M1, and `modelspec train --backend vulkan` is refused
with `E-TARGET-001`.

### Added

- A Cargo workspace of `modelspec-core`, `modelspec-runtime` and `modelspec-cli` (story:m0-workspace-foundation)
- `modelspec-core` has no Vulkan dependency under any feature or target; `task core-no-vulkan` holds it
- `modelspec_core::syntax::parse` and `syntax::print`: lexer and parser for the M0 subset (story:m0-parse-subset)
- Syntax errors `E-SYNTAX-001` to `E-SYNTAX-021`, each with line and column
- `modelspec_core::types::check_source`: name resolution, type and shape checking into `hir::Hir` (story:m0-check-types-shapes)
- Name errors `E-NAME-001` to `E-NAME-008` and type and shape errors `E-TYPE-001` to `E-TYPE-017`
- `modelspec_core::ir::tensor::TensorProgram` and `TensorProgramBuilder`: the tensor SSA IR (story:m0-tensor-ir-core)
- `TensorOp` registry: `Add`, `Sub`, `Mul`, `MatMul`, `Relu`, `Transpose`, `ReduceSum`, `ReduceMean`, `Mse`
- `ir::verify::verify` refuses a use before definition, an undeclared `ParamId` and a shape mismatch
- `ir::print::print`: a deterministic text form of the Tensor IR
- `modelspec_core::lower::lower`: Typed HIR to Tensor IR (story:m0-lower-tensor-ir)
- `modelspec_core::passes::dce::eliminate`: dead-code elimination over the Tensor IR
- `modelspec_core::autodiff::differentiate`: reverse-mode autodiff into a `TrainingProgram` (story:m0-reverse-mode-autodiff)
- `ir::training::attach_sgd` and `ir::verify::verify_step`: the verified SGD training step
- `modelspec_core::compile::compile_training`: source to a verified `CompiledTraining` in one call
- `modelspec_runtime::cpu::evaluate`: the CPU reference interpreter over Tensor IR, in `f32` and `f64` (story:m0-cpu-interpreter)
- `modelspec_runtime::train::runner::run`: seeded SGD training on the CPU interpreter (story:m0-cpu-sgd-training)
- `train::init`: Philox4x32-10 parameter initialization keyed by seed and canonical parameter path (DESIGN.md § 2.3)
- `train::data::synthetic`: M0 synthetic teacher data; data errors `E-DATA-001` to `E-DATA-003`
- `modelspec check <file>`: parse and check; diagnostics on stderr, exit 1
- `modelspec ir <file> --stage tensor|training`: print the verified Tensor IR or training step
- `modelspec train <file> --backend cpu`: train on the CPU reference, printing initial params, losses and final params
- CLI exit codes: 0 success, 1 diagnostics or runtime error, 2 usage error
- `examples/linear_regression.mspec` checks and trains; `task check-examples` runs `modelspec check` on it
- ESS contracts for the compiled artifact, checkpoint, device profile and training run in `spec/`

[Unreleased]: https://github.com/beyond10x/modelspec/compare/0.1.1...HEAD
[0.1.1]: https://github.com/beyond10x/modelspec/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/beyond10x/modelspec/releases/tag/0.1.0
