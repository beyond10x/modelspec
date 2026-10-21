# ModelSpec

A Rust compiler and runtime for `.mspec` tensor-program specifications. Humans start at
[README.md](README.md); the design is [DESIGN.md](DESIGN.md).

## State

Milestone M0 in progress (DESIGN.md revision 0.3, § 1 and § 11, iteration 1). A Cargo workspace of
three crates exists: `modelspec-core` (compiler, no Vulkan), `modelspec-runtime` (CPU oracle and
training; the `vulkan` feature is empty until M1) and `modelspec-cli` (binary `modelspec`). Module
files are declared and mostly empty; `check`, `ir` and `train` print "not implemented in this build"
and exit 1. No CI.

## Commands

Entry points are the `Taskfile.yml` tasks (go-task):

- `task build`: `cargo build --workspace`.
- `task test`: `cargo test --workspace`.
- `task lint`: `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings`.
- `task core-no-vulkan`: `cargo test -p modelspec-core --test no_vulkan`, which fails when the
  dependency graph of `modelspec-core` names a Vulkan crate on any target with all features on.
- `task ci`: all of the above; the gate.

CLI exit codes: 0 success, 1 diagnostics or runtime error, 2 usage error.

## Planning and specification

- Work is planned in the AEP store under `.engineering/` through the `aep` CLI only
  (`aep plan artifact list` to start). Never edit a store file by hand.
- Spec-first: every data contract between compiler, runtime and files on disk (compiled artifact,
  checkpoint, device profile, training-run record) is typed in `spec/` as ESS before code or stories
  use it. Run `ess specify validate --path .` from inside `spec/` after each edit.
  `spec/ess-inputs.yaml` pins the `ess` release; an older installed `ess` delegates to the pin only
  when started inside `spec/`, and refuses `--path spec` from the repository root.
- The `.mspec` language (grammar, types, shapes, gradient rules) is not an ESS domain; DESIGN.md
  § 2–4 and the tests of DESIGN.md § 9 hold it.
- Generated output (`ess generate`) goes outside `spec/`, beside it, never inside it.

## Rules

- DESIGN.md is the authority. A change to semantics, IR levels, the operator registry or an ADR in
  DESIGN.md § 12 is a design change: edit DESIGN.md in the same change and bump its revision line.
- Keep the seven invariants in DESIGN.md § 0. A change that breaks one is refused, not worked around.
- `modelspec-core` must build and test without Vulkan installed and has no Vulkan dependency under
  any feature or target. Vulkan code sits behind the `vulkan` Cargo feature of `modelspec-runtime`
  (DESIGN.md § 7).
- No Python, PyTorch, CUDA API or CUDA toolkit dependency, in code, tests or tooling.
- Correctness before speed: the CPU oracle and finite-difference gradient checks (DESIGN.md § 9,
  gates 1–3) exist before any GPU kernel is optimized.
- The files in `examples/` are target syntax. Keep them in step with DESIGN.md § 2 until a parser
  can check them, then make `modelspec check` on every example part of the gate once the parser
  supports its constructs: `linear_regression.mspec` from M0, `tiny_mlp.mspec` (modules, generics,
  `vulkan(0)`) from DESIGN.md § 11 iteration 3.
- Use `Taskfile.yml` (go-task) for build and test entry points once code exists; not a Makefile.
