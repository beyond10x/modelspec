# ModelSpec

A Rust compiler and runtime for `.mspec` tensor-program specifications. Humans start at
[README.md](README.md); the design is [DESIGN.md](DESIGN.md).

## Serves

- **O6 — self-improvement, built into all of it.** Every compile and training run leaves a typed
  record (artifact and source hashes, the deterministic setting, run steps and resume checkpoint in
  `spec/`), so a change to the compiler can be re-measured against the same run before and after.

## State

Milestone M0 implemented (DESIGN.md revision 0.6, § 1 and § 11, iteration 1; wave `m0` closed);
M1 (`epic:m1-vulkan-execution`) is next. A Cargo workspace of five crates: `modelspec-core`
(parser, checker, Tensor IR, autodiff, lowering, DCE; no Vulkan), `modelspec-runtime` (CPU oracle
and SGD training; the `vulkan` feature is empty until M1), `modelspec-observatory` (the typed
`RunEvent`s of a run and their Server-Sent Events stream; its JSON Schema under `schema/` is
`ess generate` output), `modelspec-cli` (binary `modelspec`) and `modelspec-docs` (repository
tooling: the derived pages of the documentation site).
`check` resolves names and checks types and shapes;
`ir <file> --stage tensor|training` prints the verified Tensor IR or the verified training step;
`train <file> --backend cpu` trains on the CPU reference (DESIGN.md § 8), and `--backend vulkan`
is refused with E-TARGET-001 until M1. CI: `check.yml` runs `task ci`; `shared-gates.yml` runs the
common source gates; `pages.yml` and `b10x-docs-site.yml` build and publish the documentation site.

## Commands

Entry points are the `Taskfile.yml` tasks (go-task):

- `task build`: `cargo build --workspace`.
- `task test`: `cargo test --workspace`.
- `task lint`: `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings`.
- `task core-no-vulkan`: `cargo test -p modelspec-core --test no_vulkan`, which fails when the
  dependency graph of `modelspec-core` names a Vulkan crate on any target with all features on.
- `task check-examples`: `modelspec check examples/linear_regression.mspec`;
  `examples/tiny_mlp.mspec` joins it when modules and generics land (DESIGN.md § 11 iteration 3).
- `task docs-check`: `modelspec-docs generate --check`, which fails when a generated page of the
  documentation site differs from what the code produces now; `task docs-generate` rewrites them.
- `task spec-validate`: `ess specify validate --path .` from inside `spec/`.
- `task schema-check`: `ess generate --kind schema --check` from inside `spec/`, which fails when
  a file the specification generates under `crates/modelspec-observatory/schema/` differs from it
  or is missing; an extra file there passes. `task schema-generate` rewrites them. A change to any file under `spec/` changes the schema's
  provenance digest, so run `task schema-generate` in the same change.
- `task ci`: all of the above; the gate.
- `task website`: `npm ci` and `npm run build` in `website/` (Node 22; nothing else uses Node).
- `task release-check TAG=<version>`: refuses unless the tag is a bare version, equals
  `[workspace.package] version` in `Cargo.toml`, and `CHANGELOG.md` has `## [<version>] - <date>`.

CLI exit codes: 0 success, 1 diagnostics or runtime error, 2 usage error.

## Releases

ModelSpec releases from source at bare-version tags (`0.1.0`, no `v`); every crate inherits the
workspace version and nothing is on a registry.

- `CHANGELOG.md` follows Keep a Changelog 1.1.0. Changes collect under `## [Unreleased]`, one line
  per change a user sees, naming the public item in backticks and its story id where it fits.
- A release commit on `main` sets `[workspace.package] version` in `Cargo.toml` and `Cargo.lock`,
  turns `## [Unreleased]` into `## [<version>] - <date>` under a new empty `## [Unreleased]`, and
  moves the link references at the bottom. `task release-check TAG=<version>` passes on it.
- `check` and `Shared source gates` are green on that commit before it is tagged.
- The tag is the bot's, on that commit:
  `b10x-gates bot --repository beyond10x/modelspec -- tag -a <version> -m "ModelSpec <version>" <commit>`,
  then `b10x-gates bot --repository beyond10x/modelspec -- push origin <version>`.
- `.github/workflows/release.yml` runs on the tag: `release-check`, then `check.yml`, then it builds
  the `modelspec` binary for Linux x86_64 and creates the GitHub Release with the version's
  CHANGELOG section as notes, `modelspec-<version>-x86_64-unknown-linux-gnu` and its `.sha256`.
- A release is finished when the tag, the green `release` run and the Release with both assets are
  verified. Never move or delete a pushed tag; fix forward with the next version.
- Atlas is never on this path. No push, PR landing, release or docs deploy here waits for an Atlas
  pipeline or an Atlas PR; catalog updates in Atlas land on their own schedule afterwards.

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
- `website/docs/reference/`, `website/docs/examples/` and `website/data/` are generated by
  `modelspec-docs`: change the code, then `task docs-generate`; never edit them by hand. The other
  pages under `website/docs/` are hand-written; update them in the change that alters what they
  describe. The pitch deck is `docs/pitch/index.html`, which the site build serves at
  `/modelspec/pitch/`; never copy it into `website/`.
