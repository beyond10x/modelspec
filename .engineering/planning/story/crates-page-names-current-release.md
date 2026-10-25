---
format: aep.planning-md/3
id: story:crates-page-names-current-release
kind: story
status: implemented
title: The crates page names the current release
relations:
- serves: vision:modelspec
scope:
- confidence: cited
  path: crates/modelspec-docs/src/crates.rs
- confidence: cited
  path: website/docs/reference/crates.md
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-10-25T12:36:15Z", actor: "human:timo", revision: 3, decided_on: {"recorded":{"test_result":1}}}
- {from: "proposed", to: "active", at: "2026-10-25T12:36:15Z", actor: "human:timo", revision: 4, decided_on: {"recorded":{"test_result":1}}}
- {from: "active", to: "implemented", at: "2026-10-25T12:36:16Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# The crates page names the current release

## Why

Review of 0.5.0 (finding D2), still open at main 496e98e. The generated crates page says the
"0.1.0 release" ships the `modelspec` binary (`crates/modelspec-docs/src/crates.rs:150`,
`website/docs/reference/crates.md:10`), a literal that has been wrong since 0.1.1.

Outcome: the page names the release the workspace version is.

## Acceptance

- **version-from-metadata**: the release named and linked on the page comes from `cargo metadata`,
  not a literal.
- **stale-page-fails**: `task check` fails when `website/docs/reference/crates.md` names a version
  other than the workspace's.
- `task ci` exits 0.

## Scope

- `crates/modelspec-docs/src/crates.rs`
- `website/docs/reference/crates.md`

## Out of scope

- Other generated pages.
