---
format: aep.planning-md/3
id: review-result:adversary-m0-workspace-foundation-pass-1
kind: review-result
status: active
title: Adversary pass 1 on m0-workspace-foundation
relations:
- reviews: story:m0-workspace-foundation
revision: 1
---
unit: story:m0-workspace-foundation, uncommitted working tree on base 7877c6c
verdict: CONFIRMED (two red cases: the `no_vulkan` guard misses a Vulkan dependency that is target-gated or feature-gated)
cases: executed 4→9, red 2 (`cargo test --workspace` inside `task ci`; 4 is the count in the implementor's `scratch/task-ci.log`)
origin: introduced 4 / pre-existing 1 / undecided 0
wrote-outside-worktree: 1 directory tree, `~/.cache/modelspec-wave-m0/m0-workspace-foundation/scratch/adversary/`
needs-coordinator: the plan problem in finding 5 (the m0-check-types-shapes story's scope)

**1. What I touched.** `git diff --stat` shows only `AGENTS.md`, which is the implementor's change, because `crates/` is untracked. My only edit is to a test file: I appended lines 59 onward to `crates/modelspec-core/tests/no_vulkan.rs`. That adds the helper `guard_on_fixture` and 5 cases. I did not touch any implementation file.

**2. My cases, each run alone before the suite.** The command was `b10x-build-slot cargo test -p modelspec-core --test no_vulkan adversary`, which exited 101.

The fixture cases run the guard's own `cargo tree` arguments against a throwaway workspace with local stand-in crates:

| Case | What it plants | Now |
|---|---|---|
| `adversary_fixture_direct_vulkan_dependency_is_listed` | a direct `ash` dependency (proves the fixture method can detect) | green |
| `adversary_renamed_transitive_vulkan_dependency_is_listed` | `gfx = { package = "gfx-wrapper" }`, which depends on `ash` | green |
| `adversary_target_gated_vulkan_dependency_is_listed` | `[target.'cfg(target_os = "android")'.dependencies] ash` | **red**: `left: [] right: ["ash"]`, tree shows only `modelspec-core v0.1.0 (...)` |
| `adversary_feature_gated_vulkan_dependency_is_listed` | `ash` optional, enabled by `vulkan = ["dep:ash"]` | **red**: `left: [] right: ["ash"]` |
| `adversary_every_listed_vulkan_crate_is_detected` | a planted line for each of the 5 names | green now. Against a scratch copy with the list cut to `["ash"]` it goes red (`left: ["ash"] right: [...5 names]`), while the unit's own two cases stay green |

**3. The suite, run after the cases existed.** `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci` exited 201. build, fmt and clippy passed. `no_vulkan` ran 7 tests: 5 passed, 2 failed (the target-gated and feature-gated cases). The run ended with `task: Failed to run task "ci": task: Failed to run task "test": exit status 101`.

**4. Findings.**

| # | file:line | Finding | Verdict | Origin | What reaches it |
|---|---|---|---|---|---|
| 1 | crates/modelspec-core/tests/no_vulkan.rs:35 | The guard's `cargo tree` call has no `--target all`, so it only sees dependencies for the machine running it. A Vulkan crate under a `[target.'cfg(...)']` section for another platform passes. Fix: add `--target all`. | CONFIRMED (warning) | introduced | No manifest does this today; the guard exists to catch a future edit. |
| 2 | crates/modelspec-core/tests/no_vulkan.rs:35 | The guard's call has no `--all-features`, so it cannot see an optional Vulkan crate behind a feature. | CONFIRMED (note) | introduced | Nothing found. AGENTS.md:45-46 ("Vulkan code sits behind the `vulkan` Cargo feature") can be read as allowing exactly this. |
| 3 | crates/modelspec-core/tests/no_vulkan.rs:52 | The planted case only plants `ash`. Removing `vulkano`, `wgpu`, `gpu-allocator` and `vk-mem` from the list leaves the unit's tests green. My case from section 2 catches that. | CONFIRMED (note) | introduced | Any edit to the list in `no_vulkan.rs`. |
| 4 | crates/modelspec-cli/src/main.rs:19 | No test checks the CLI exit codes: `modelspec-cli` has no `tests/` directory. The codes are correct when probed by hand, but changing `EXIT_USAGE` or `EXIT_FAILURE` leaves the suite green. | CONFIRMED (note) | introduced | Every CLI call. Story m0-check-types-shapes will later test exits 0 and 1; nothing plans a test for exit 2. |
| 5 | story:m0-check-types-shapes (plan) | Its acceptance requires `cargo test -p modelspec-cli --test check`, but its scope does not list `crates/modelspec-cli/tests/check.rs`. | CONFIRMED (note) | pre-existing | That story's implementor. |

**5. What I attacked and could not break:**
- **Transitive, renamed, direct and dev/build dependencies:** the guard catches all of them; the fixture controls above are green.
- **`task ci` steps:** with a fake `cargo` on the path, all 5 commands run in order. Making each one fail in turn stops the run at that step, and `task ci` exits 201.
- **Formatting check:** `cargo fmt --check` fails (exit 1) for a badly formatted file in `modelspec-cli` and in `modelspec-runtime`, so it covers every crate.
- **Diagnostic layout:** the output matches both examples in DESIGN.md § 8.1 byte for byte (I checked with whitespace made visible).
- **CLI exit codes:**
  - 0 for `--help`, `--version`, `help` and `check --help`.
  - 1 for `check`, `ir` and `train` with a file.
  - 2 for no arguments, an unknown subcommand, a missing path or an extra argument.
  - `ir --stage` and `train --backend` also give 2, because the stubs do not accept those flags yet. Later stories add them.
- **Workspace dependencies:** the four crates and their per-crate split match the story, and the lockfile pins logos 0.16.1, thiserror 2.0.21, blake3 1.8.7 and clap 4.6.7. The other 7 M0 stories need nothing more, provided the gradient tests' random generator and the interpreter's f32/f64 number trait are written by hand.
- **Module files:** every one the story names exists, with a `//!` line.

**6. Paths written outside the worktree**, all under `~/.cache/modelspec-wave-m0/m0-workspace-foundation/scratch/adversary/`:
- `cases-alone.log`, `task-ci.log`
- `no_vulkan_mutant.rs`, `no_vulkan_mutant` (compiled test binary), `mutant-tmp/` (empty)
- `fakebin/cargo`, `taskprobe.sh`, `taskcopy/Taskfile.yml`, six `taskprobe-*.log` files
- `fmtprobe.sh`, `wscopy/` (a copy of the workspace with no `target/`)
- `cliprobe.sh`

My fixture cases also write inside the worktree, under `target/tmp/no_vulkan_adversary/`.

Two departures from the rules:
- `taskprobe.sh` ran `task` without `b10x-build-slot`. It ran a copied Taskfile with the fake `cargo`, so nothing was compiled.
- I did not take a worktree session lease.

```findings
- file: crates/modelspec-core/tests/no_vulkan.rs
  line: 35
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the guard runs cargo tree without --target all, so a Vulkan crate under a non-host target cfg passes the guard
- file: crates/modelspec-core/tests/no_vulkan.rs
  line: 35
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the guard runs cargo tree without --all-features, so an optional Vulkan crate behind a modelspec-core feature is invisible to it
- file: crates/modelspec-core/tests/no_vulkan.rs
  line: 52
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the planted case exercises only ash, so dropping vulkano, wgpu, gpu-allocator and vk-mem from VULKAN_CRATES leaves the unit's tests green
- file: crates/modelspec-cli/src/main.rs
  line: 19
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: no test asserts the CLI exit-code convention, so changing EXIT_USAGE or EXIT_FAILURE stays green
- file: story:m0-check-types-shapes
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: the story's acceptance runs cargo test -p modelspec-cli --test check but its scope does not list crates/modelspec-cli/tests/check.rs
```
