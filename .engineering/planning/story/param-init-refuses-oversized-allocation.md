---
format: aep.planning-md/3
id: story:param-init-refuses-oversized-allocation
kind: story
status: draft
title: Parameter init refuses an allocation it cannot make
scope:
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/init.rs
revision: 2
---
# Parameter init refuses an allocation it cannot make

## Why

Review of 0.5.0 (finding C5), still open at main 496e98e. Parameter initialisation and the data
readers allocate the element count a declaration asks for without a fallible allocation
(`crates/modelspec-runtime/src/train/init.rs:123-145`,
`crates/modelspec-runtime/src/train/data.rs:452-471`). A declaration too large for the host aborts
the process instead of reporting an error.

Outcome: a parameter or input the host cannot allocate is a named diagnostic and a non-zero exit,
not an abort.

## Acceptance

- **oversized-param-named-error**: `param w: Tensor<f32,[65536,65536]>` exits with a named `E-`
  code and the parameter's path, and no abort signal.
- **oversized-input-named-error**: a dense input whose shape cannot be allocated exits the same way.
- DESIGN.md names the code.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/train/init.rs`
- `crates/modelspec-runtime/src/train/data.rs`

## Out of scope

- Device-memory exhaustion on Vulkan, which `vkAllocateMemory` already reports.
- Bounds on input files, which story:input-readers-bound-before-allocating holds.
