# Unit brief: story:cpu-matmul-row-order

## Identity

```
story:  story:cpu-matmul-row-order
branch: impl/cpu-matmul-row-order       forked from main
base:   ab9ed275eb5f6ef12367d968e3ebf76de7958e76 (release 0.1.1 plus a planning commit; no code change since 0.1.1)
```

The story is not in this worktree's store. Read it in full first from the planning tree:
`aep plan artifact show story:cpu-matmul-row-order`, run inside
`~/.local/state/worktree/trees/b10x/modelspec/modelspec-plan-ui` (read only there). Its
`## Acceptance` and `## Scope` sections are the contract. Also read
`aep plan artifact show verification-report:mnist-benchmark-0-1-1` and
`aep plan artifact show epic:pytorch-10x-mnist` there for why the speed matters.

Review your own change with `git diff ab9ed27` plus `git status` (you do not commit).

## The triple

```
worktree:  ~/.local/state/worktree/trees/b10x/modelspec/modelspec-impl-cpu-matmul-row-order
build dir: <worktree>/target
scratch:   ~/.cache/modelspec-wave-cpu-perf-1/cpu-matmul-row-order/scratch
```

## The file assignment

| | |
|---|---|
| **yours** | `crates/modelspec-runtime/src/cpu/interpreter.rs`, new test files under `crates/modelspec-runtime/tests/`, `crates/modelspec-runtime/benches/` (new), `crates/modelspec-runtime/Cargo.toml` (bench entry, dev-dependencies if any), `Cargo.lock` (only as cargo changes it), `CHANGELOG.md` (one line under `## [Unreleased]`) |
| **not yours** | `DESIGN.md`, every other crate, the CLI goldens under `crates/modelspec-cli/tests/golden/` (they must pass unchanged; never re-pin them), everything under `.engineering/`. If you need a change there, write the patch to `<scratch>/<name>.patch` and name it in the report |

## Measuring the speed (acceptance 3)

The branch is the 0.1.1 code, so the baseline is this tree before your interpreter change:

1. Write the benchmark target first (784 → 64 ReLU → 10, no bias, f32, MSE, SGD lr 0.1, batch 100,
   synthetic seeded data, a fixed number of steps; prints images/s). Not part of `task ci`.
2. Run it 3 times on the unchanged interpreter, pinned with `taskset -c 3`, through
   `b10x-build-slot`, release profile. Save the output to `<scratch>/bench-base-<n>.log`.
3. Make the change; run the same command 3 times; save `<scratch>/bench-change-<n>.log`.
4. Report both medians and the ratio. The host is shared (load average 18 to 30 today); quote
   `/proc/loadavg` before each set. If the ratio is under 3, report the number; do not tune past
   the story's Out of scope (no threads, no SIMD intrinsics, no k-blocking, no FMA).

Also report where the step time goes before and after (for example with `perf stat` or a timer
around the interpreter's op dispatch), if that is cheap: the next story for the CPU lane is chosen
from it.

## Repository invariants

* `AGENTS.md` at the repository root is binding; read it first.
* No Python, PyTorch, CUDA dependency in code, tests or tooling. No Vulkan dependency in
  `modelspec-core`.
* No comments in config or manifest files (`Cargo.toml`, `Taskfile.yml`, workflow YAML).
* No absolute home-directory paths in any tracked file.
* Nothing naming any external company, customer or product outside this repository.
* Each worktree builds into its own `target/` inside the worktree; never set `CARGO_TARGET_DIR`.
* Every cargo command runs through `b10x-build-slot`, for example
  `b10x-build-slot cargo test -p modelspec-runtime`. Debug test runs use
  `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only`.
* `cargo fmt -p <crate>`, never `cargo fmt --all`.
* Never write under `.engineering/`, and no `aep plan artifact` write verb in any tree; reads are fine.
* No `git worktree`, `git commit`, `git add`, `git stash` or branch command; leave changes in the
  working tree.
* Nothing under `/tmp` or `/dev/shm`; no heredoc feeding a stdin reader; write scripts to scratch
  and run them by path.
* Never delete a `target/` directory.
* Never call the GitLab API with a personal token.
* No GPU or Vulkan runs.

## The gate (package-scoped)

```
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot cargo test -p modelspec-runtime -p modelspec-cli
b10x-build-slot cargo clippy -p modelspec-runtime --all-targets -- -D warnings
cargo fmt -p modelspec-runtime -- --check
b10x-build-slot task docs-check
```

Run each before reporting and quote each summary line verbatim, with its own exit status. Test
first: write the bit-equality test (acceptance 2) and the planted-defect check (acceptance 4) and
show them red where the story says they must be red, paste the red run's summary in the report,
then implement.

## The report header

Six lines, before any prose:

```
unit: story:cpu-matmul-row-order
verdict: <green|red|blocked>
cases: executed <before>→<after>, red <n>
origin: n/a
wrote-outside-worktree: <paths, or none>
needs-coordinator: <what you could not do and who owns it, or none>
```

After the header: the bench table (base and change, 3 runs each, median, ratio), the files
changed, and the acceptance scenarios mapped to test names.
