---
format: aep.planning-md/3
id: story:crates-page-names-current-release
kind: story
status: draft
title: The crates page names the current release
scope:
- confidence: cited
  path: crates/modelspec-docs/src/crates.rs
- confidence: cited
  path: website/docs/reference/crates.md
revision: 2
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
