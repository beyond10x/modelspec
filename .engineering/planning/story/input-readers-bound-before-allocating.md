---
format: aep.planning-md/3
id: story:input-readers-bound-before-allocating
kind: story
status: draft
title: Input-file readers check what they read before they allocate
revision: 1
---
# Input-file readers check what they read before they allocate

## Why

Code review of wave m1-h, findings 2–5 (warnings, CONFIRMED), names four readers of input files
that allocate before they check the file. The review is kept outside the repository; its details
go into this story when the fix is released.

## Acceptance

- Each of the four readers named in the review checks the file's type and size against a stated
  bound before it allocates, and refuses past it with the reader's existing error code.
- One host-only test per reader reaches the refusal.
- DESIGN.md states each bound.
- `task ci` exits 0.
