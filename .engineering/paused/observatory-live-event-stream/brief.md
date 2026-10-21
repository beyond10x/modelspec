# Unit brief: story:observatory-live-event-stream

## Identity

```
story:  story:observatory-live-event-stream
branch: impl/observatory-live-event-stream       forked from wave/observatory-1
base:   ab9ed275eb5f6ef12367d968e3ebf76de7958e76
```

Read the story in full first: `aep plan artifact show story:observatory-live-event-stream`
(run inside the worktree). Its `## Acceptance` and `## Scope` sections are the contract. Review
your own change with `git diff ab9ed27...HEAD` plus `git status` (you do not commit).

## The triple

```
worktree:  ~/.local/state/worktree/trees/b10x/modelspec/modelspec-impl-observatory-live-event-stream
build dir: <worktree>/target
scratch:   ~/.cache/modelspec-wave-observatory-1/observatory-live-event-stream/scratch
```

## The file assignment

| | |
|---|---|
| **yours** | `crates/modelspec-observatory/` (new crate), `crates/modelspec-runtime/src/train/runner.rs`, `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/Cargo.toml`, `crates/modelspec-cli/tests/`, `Cargo.toml`, `Cargo.lock`, `Taskfile.yml`, `.github/workflows/check.yml`, `spec/domains/observatory.yaml`, `spec/domains/training.yaml` (only if the run lifecycle settlement needs it), `website/docs/reference/cli.md` and `crates.md` (only via `task docs-generate`), `AGENTS.md` (State line: crate count and the new crate), `CHANGELOG.md` (one line under `## [Unreleased]`) |
| **not yours** | `DESIGN.md` (another unit owns it in this wave; coordinator decision D1: § 7 is a starting sketch and is not updated for the new crate), `crates/modelspec-core/src/autodiff.rs`, any `input_adjoint.rs`, everything under `.engineering/`. If you need a change there, write the patch to `<scratch>/<name>.patch` and name it in the report |

Coordinator decision D2: `task ci` gains `ess specify validate` (run from inside `spec/`); add an
`ess` install step to `.github/workflows/check.yml` pinned to the release `spec/ess-inputs.yaml`
requires (read it there; do not hard-code a different version). Use a prebuilt release download
with checksum verification, not a cargo build from source.

Settle the UNMAPPED markers the story names in `spec/domains/observatory.yaml` (and the run
lifecycle in `training.yaml` if the story requires it). Run `ess specify validate --path .` from
inside `spec/` after each edit and quote its last line. Generated output (the RunEvent JSON
Schema) goes outside `spec/`, as the story says.

## Repository invariants

* `AGENTS.md` at the repository root is binding; read it first.
* Running code is Rust. Browser code (not in this unit) would be Vue 3 + TS + Vite + pnpm.
* No Python, PyTorch, CUDA dependency in code, tests or tooling. No Vulkan dependency in
  `modelspec-core`.
* No comments in config or manifest files (`Cargo.toml`, `Taskfile.yml`, workflow YAML, spec YAML
  beyond the existing UNMAPPED markers you resolve or keep).
* No absolute home-directory paths in any tracked file.
* Nothing naming any external company, customer or product outside this repository.
* Each worktree builds into its own `target/` inside the worktree; never set `CARGO_TARGET_DIR`.
* Every cargo command runs through `b10x-build-slot`, for example
  `b10x-build-slot cargo test -p modelspec-observatory`. Full test runs use
  `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only`.
* `cargo fmt -p <crate>`, never `cargo fmt --all`.
* Never write under `.engineering/`, and no `aep plan artifact` write verb; reads are fine.
* No `git worktree`, `git commit`, `git add`, `git stash` or branch command; leave changes in the
  working tree.
* Nothing under `/tmp` or `/dev/shm`; no heredoc feeding a stdin reader; write scripts to scratch
  and run them by path.
* Never delete a `target/` directory.
* Never call the GitLab API with a personal token.

## The gate (package-scoped)

```
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot cargo test -p modelspec-observatory -p modelspec-runtime -p modelspec-cli
b10x-build-slot cargo clippy -p modelspec-observatory -p modelspec-runtime -p modelspec-cli --all-targets -- -D warnings
cargo fmt -p modelspec-observatory -p modelspec-runtime -p modelspec-cli -- --check
b10x-build-slot task docs-check
(cd spec && ess specify validate --path .)
```

Run each before reporting and quote each summary line verbatim, with its own exit status. Test
first: write the failing tests for the acceptance scenarios, run them red, paste the red run's
summary in the report, then implement.

## The report header

Six lines, before any prose:

```
unit: story:observatory-live-event-stream
verdict: <green|red|blocked>
cases: executed <before>→<after>, red <n>
origin: n/a
wrote-outside-worktree: <paths, or none>
needs-coordinator: <what you could not do and who owns it, or none>
```

After the header: a confirmation table of the story's `## Scope` lines (each inferred line
confirmed or corrected, with the path), the files changed, and the acceptance scenarios mapped to
test names.
