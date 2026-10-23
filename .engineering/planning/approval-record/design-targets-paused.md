---
format: aep.planning-md/3
id: approval-record:design-targets-paused
kind: approval-record
status: draft
title: design/targets stays a paused draft (orchestrator, 2026-10-23)
relations:
- serves: vision:modelspec
revision: 1
---
# design/targets stays a paused draft

Decided by the orchestrator on 2026-10-23, after Timo asked it to take this session's decisions
(Timo, verbatim: "ask chef to adopt you, chef makes decissions").

## The question

The remote branch `design/targets` (head `a793cef`, 2026-10-21, "design: compile targets survey and
x86-64 experiment (draft, paused)", +1413 lines in `docs/design/targets.md` and
`docs/design/targets-research-notes.md`) is not on `main`. Options: A keep it as a paused draft,
B open it as its own pull request, C delete it.

## The decision

b10x-chef-orchestrator, verbatim: "design/targets: option A. Keep it on the remote as a paused
draft. Do not open a PR and do not delete it. Reasons: deleting an unmerged branch is outside the
grant and Timo's call alone, and a compile-targets PR is work outside wave m1-h. If it should ever
proceed, it gets its own story in your store first."

## What it grants

Nothing. The branch stays as it is; any further work on it starts with a story in this store.
