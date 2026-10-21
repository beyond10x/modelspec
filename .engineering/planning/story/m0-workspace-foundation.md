---
format: aep.planning-md/3
id: story:m0-workspace-foundation
kind: story
status: active
title: 'Workspace foundation: three crates, Taskfile, a Vulkan-free core'
relations:
- decomposes: epic:m0-semantic-core
- serves: vision:modelspec
scope:
- confidence: cited
  path: AGENTS.md
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: Cargo.toml
- confidence: cited
  path: Taskfile.yml
- confidence: cited
  path: crates/modelspec-cli/Cargo.toml
- confidence: cited
  path: crates/modelspec-cli/src/main.rs
- confidence: cited
  path: crates/modelspec-core/Cargo.toml
- confidence: cited
  path: crates/modelspec-core/src/diag.rs
- confidence: cited
  path: crates/modelspec-core/src/lib.rs
- confidence: cited
  path: crates/modelspec-core/tests/no_vulkan.rs
- confidence: cited
  path: crates/modelspec-runtime/Cargo.toml
- confidence: cited
  path: crates/modelspec-runtime/src/lib.rs
revision: 16
transitions:
- {from: "draft", to: "proposed", at: "2026-10-21T09:12:24Z", actor: "human:timo", revision: 15, decided_on: {"recorded":{"review_outcome":2}}}
- {from: "proposed", to: "active", at: "2026-10-21T09:12:25Z", actor: "human:timo", revision: 16, decided_on: {"recorded":{"review_outcome":2}}}
---
# Workspace foundation: three crates, Taskfile, a Vulkan-free core

## Context

The repository holds a design and no code. Every other M0 story lands inside the Cargo workspace
this story creates. It is the prefactoring the other M0 stories `depends_on`: it fixes the crate
layout, the module files, the diagnostics type and the build entry points once, so later stories
add code to existing files instead of contending for `lib.rs`, `main.rs` and `Taskfile.yml`.

Implements DESIGN.md § 7 (workspace of three crates; the core compiles and tests without Vulkan;
a `vulkan` Cargo feature gates backend code) and the AGENTS.md rules on `Taskfile.yml` and on
`modelspec-core` building without Vulkan. Owns the epic in-scope line "Workspace of three crates;
`modelspec-core` builds and tests without Vulkan".

## Lands in

- `Cargo.toml` (workspace root: members, `resolver`, shared `[workspace.package]` with
  `license = "Apache-2.0"`, edition 2024, shared `[workspace.dependencies]` naming every crate M0
  uses: `logos`, `thiserror`, `blake3`, `clap`).
- Every M0 crate dependency is declared here, once, so no later M0 story edits a `Cargo.toml` or
  `Cargo.lock` and stories that run at once do not collide on them: `modelspec-core` depends on
  `logos` and `thiserror`; `modelspec-runtime` on `modelspec-core`, `thiserror` and `blake3`;
  `modelspec-cli` on both libraries and `clap`. A later story that needs another crate says so in its
  Lands in.
- `crates/modelspec-core/`: `Cargo.toml`; `src/lib.rs` declaring every M0 module as an empty file
  with a one-line `//!` doc: `syntax/{mod,lexer,parser,ast}.rs`, `resolve.rs`, `types.rs`, `hir.rs`,
  `ops.rs`, `ir/{mod,tensor,training,verify,print}.rs`, `autodiff.rs`, `lower.rs`,
  `passes/{mod,dce}.rs`, `compile.rs`, and `diag.rs`. Later stories fill these files and do not edit
  `lib.rs`.
- `crates/modelspec-core/src/diag.rs`: the user-facing diagnostic type (code such as `E-TYPE-012`,
  message, primary `SourceSpan` with file, line and column, labelled detail lines) and its plain-text
  rendering in the layout of DESIGN.md § 8.1 (first line `<code>: <message>`, then indented
  `<path>:<line>:<col>`, then the detail lines). Codes are declared next to the stage that emits them,
  not in this file.
- `crates/modelspec-runtime/`: `Cargo.toml` with features `default = []` and `vulkan = []` (empty in
  M0); `src/lib.rs` declaring `cpu/{mod,tensor,interpreter}.rs` and
  `train/{mod,optimizer,runner,data,init}.rs` as empty files.
- `crates/modelspec-cli/`: binary `modelspec` (clap) with subcommands `check`, `ir`, `train`, each in
  its own file `src/{check,ir,train}.rs`, each printing "not implemented in this build" and exiting 1.
  `src/main.rs` fixes the exit-code convention: 0 success, 1 diagnostics or runtime error, 2 usage.
- `Taskfile.yml` (go-task, not a Makefile) with tasks `build`, `test` (`cargo test --workspace`),
  `lint` (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`),
  `core-no-vulkan` and `ci` (all of them).
- `AGENTS.md` § State and a short commands list pointing at the Taskfile tasks.

`core-no-vulkan` runs `cargo test -p modelspec-core --test no_vulkan`. The test
`crates/modelspec-core/tests/no_vulkan.rs` holds a function that lists the Vulkan crates (`ash`,
`vulkano`, `wgpu`, `gpu-allocator`, `vk-mem`) named in `cargo tree` output, and two cases: the
output of `cargo tree -p modelspec-core -e normal,build,dev` on the committed tree lists none, and a
planted line `ash v0.38.0` is listed. The planted case is what shows the guard can fail; no manual
mutation is part of the acceptance.

## Out of scope

- Any lexer, parser, checker, IR, interpreter or training code (later M0 stories).
- Any Vulkan or SPIR-V dependency or code; the `vulkan` feature stays empty until M1.
- CI configuration (no CI exists; AGENTS.md § State).
- Top-level `tests/` and `benches/` crates. Integration tests live in `crates/<crate>/tests/`;
  `.mspec` fixtures for them live under `tests/language/` at the repository root (DESIGN.md § 7).

## Domain relations

None. The story touches no ESS-typed contract in `spec/domains/`.

## Acceptance

`task ci` exits 0 on the committed tree, which includes both cases of
`cargo test -p modelspec-core --test no_vulkan`.
