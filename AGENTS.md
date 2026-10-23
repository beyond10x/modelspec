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
and SGD training; the `vulkan` feature holds Vulkan device discovery, negotiation, SPIR-V
emission and the training runner `vulkan::runner` through `ash` and `rspirv`), `modelspec-observatory` (the typed
`RunEvent`s of a run and their Server-Sent Events stream; its JSON Schema under `schema/` is
`ess generate` output), `modelspec-cli` (binary `modelspec`) and `modelspec-docs` (repository
tooling: the derived pages of the documentation site).
`check` resolves names and checks types and shapes;
`ir <file> --stage tensor|training` prints the verified Tensor IR or the verified training step;
`train <file> --backend cpu` trains on the CPU reference (DESIGN.md § 8), and `--backend vulkan`
trains on one Vulkan device with one host wait per run plus one per requested checkpoint
(`--stats` prints its counters); `--checkpoint`, `--checkpoint-every` and `--resume` work on both
backends, and `infer <file> --backend cpu|vulkan --checkpoint <path>` runs a checkpoint's forward
program and loss, on Vulkan with one host wait; a CLI built without the `vulkan` feature refuses
`--backend vulkan` with E-TARGET-001. `inspect-device --backend vulkan`
prints each device's profile and negotiation verdict; the device tests run on a GPU only with
`--ignored`. `train --serve <addr>` streams a run's `RunEvent`s, `--record <path>` writes them to a
file and `observe <path>` replays one; with `--ui ui/dist` either also serves the observatory UI at
`/`, the Vue 3 + TypeScript app in `ui/` (Vite, pnpm) that draws the loss and every metric per step.
CI: `check.yml` runs `task ci`; `shared-gates.yml` runs the
common source gates; `pages.yml` and `b10x-docs-site.yml` build and publish the documentation site.

## Commands

Entry points are the `Taskfile.yml` tasks (go-task):

- `task build`: `cargo build --workspace`.
- `task test`: `cargo test --workspace`, then `cargo test -p modelspec-runtime --features vulkan`,
  whose `spirv_val` test runs `spirv-val --target-env vulkan1.2` (SPIRV-Tools) on every emitted
  elementwise, broadcast and SGD module and fails, not skips, without it.
- `task lint`: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  the same clippy run of `modelspec-runtime` with `--features vulkan`.
- The device tests need a Vulkan device and are ignored by default:
  `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_elementwise -- --ignored`.
  `MODELSPEC_VK_VALIDATION=1` creates the instance with the Khronos validation layer; each
  message it reports fails the test. The training run on the device is
  `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_train -- --ignored`,
  which prints µs per step and dispatches per step; `cargo test -p modelspec-cli --test train_vulkan -- --ignored`
  compares the lines of `train --backend vulkan` with the CPU run.
- `crates/modelspec-runtime/tests/no_cuda.rs` (in `task test`) fails when the runtime's
  dependency graph names a CUDA crate or a built binary links a CUDA library.
- `task core-no-vulkan`: `cargo test -p modelspec-core --test no_vulkan`, which fails when the
  dependency graph of `modelspec-core` names a Vulkan crate on any target with all features on.
- `task cli-without-vulkan`: the `inspect-device`, `train_linear` and `train_linear_adversary`
  tests of a CLI built without the `vulkan` feature, which refuses `--backend vulkan` with
  E-TARGET-001.
- `task check-examples`: `modelspec check` on `examples/linear_regression.mspec` and
  `examples/mnist_mlp.mspec`; `examples/tiny_mlp.mspec` joins it when modules and generics land (DESIGN.md § 11 iteration 3).
- `task docs-check`: `modelspec-docs generate --check`, which fails when a generated page of the
  documentation site differs from what the code produces now; `task docs-generate` rewrites them.
- `task spec-validate`: `ess specify validate --path .` from inside `spec/`.
- `task schema-check`: `ess generate --kind schema --check` from inside `spec/`, which fails when
  a file the specification generates under `crates/modelspec-observatory/schema/` differs from it
  or is missing; an extra file there passes. `task schema-generate` rewrites them. A change to any file under `spec/` changes the schema's
  provenance digest, so run `task schema-generate` in the same change.
- `task ui-types-check`: `modelspec-ui-types --check`, which fails when
  `ui/src/generated/observatory.ts` differs from the TypeScript the Rust types of
  `crates/modelspec-observatory/src/event.rs` generate (`ts-rs`); `task ui-types` rewrites it.
  Never edit `ui/src/generated/` by hand. The test `observatory_ui_types_in_sync` (in `task test`)
  makes the same comparison.
- `task ui`: in `ui/`, `pnpm install --frozen-lockfile`, `pnpm typecheck`, `pnpm test` (the
  `node --test` cases of `src/lib/`) and `pnpm build` into `ui/dist/` (Node 22.18 or later, pnpm
  11; `ui/node_modules/` and `ui/dist/` are never committed).
- `task ui-browser`: `task ui`, then the `#[ignore]`d browser tests of
  `crates/modelspec-cli/tests/observatory_ui.rs` and `observatory_ui_adversary.rs` with
  `--ignored`. They start Chrome or Chromium headless through `headless_chrome`: the executable
  named by `CHROME`, else the first of
  `google-chrome-stable`, `chromium`, `chromium-browser`, `chrome`, ... on `PATH`; `check.yml` sets
  `CHROME`.
- `task ci`: all of the above; the gate. `task ci` needs Node and pnpm for
  `task ui` and a Chrome for `task ui-browser`; `task build` and `task test` need neither, and
  `sh ui/scripts/build-without-node.sh` checks that with a `PATH` that holds no `node` or `pnpm`
  (`task ci` does not run it).
- `task website`: `npm ci` and `npm run build` in `website/` (Node 22). Node is used only here and
  by `task ui`.
- `task release-check TAG=<version>`: refuses unless the tag is a bare version, equals
  `[workspace.package] version` in `Cargo.toml`, and `CHANGELOG.md` has `## [<version>] - <date>`.

CLI exit codes: 0 success, 1 diagnostics or runtime error, 2 usage error.

## Observatory UI

Two build decisions of story:observatory-ui-live-loss-chart:

- The app lives in `ui/` at the repository root (Vue 3, TypeScript, Vite, pnpm; the workspace
  precedent is uilab's `widget/`). `ui/src/generated/` is `modelspec-ui-types` output.
- The server finds the built app at run time, not at compile time: `--ui <dir>` on
  `train --serve` and on `observe` names the directory (`ui/dist` after `task ui`), which is read
  whole into memory before the address is bound and served at `/` beside `/events`
  (`spec/domains/observatory.yaml`, `UiDirectory`). Nothing embeds the app in the binary, so no
  `build.rs` or cargo step calls Node or pnpm, and `task build` and `task test` keep working on a
  machine without them. Without `--ui` only `/events` is served.

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
  Python is allowed only for opponents (the operator, 2026-10-23: "we never want to have any
  1st-level python code in our repo, python is only allowed for opponents"). An opponent is a
  competing implementation, such as a PyTorch CUDA baseline, that a benchmark runs ModelSpec
  against. It lives under `opponents/`, and no build, test, gate or `task` target of ModelSpec
  runs it or depends on it.
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
- A copy of the tree built outside the worktree (an adversary's mutant or base copy) has its
  `target/` removed with `cargo clean --target-dir <dir>` when its unit merges. Twelve such
  targets of merged waves held 9.6 GB on 2026-10-23.
