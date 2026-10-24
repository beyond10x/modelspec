---
format: aep.planning-md/3
id: story:param-init-refuses-oversized-allocation
kind: story
status: draft
title: Parameter init refuses an allocation it cannot make
scope:
- confidence: cited
  path: DESIGN.md
- confidence: cited
  path: crates/modelspec-runtime/src/train/data.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/init.rs
revision: 4
---
# Parameter init refuses an allocation it cannot make

## Why

Review of 0.5.0 (finding C5), still open at main 496e98e. Parameter initialisation and the dense
data reader allocate the element count a declaration asks for with no fallible path.

Checked at HEAD, `param w: Tensor<f32,[65536,65536]>` does not fail through `zeros`:

- `zeros` (`crates/modelspec-runtime/src/train/data.rs:452-471`) builds only input and target
  tensors (its callers are `data.rs:93` and `data.rs:373`). It refuses only an element count that
  overflows `usize` (`data.rs:461-469`, `TooManyElements`), then allocates with `vec![0.0; count]`
  (`data.rs:470`).
- Parameters are drawn by `draw` (`crates/modelspec-runtime/src/train/init.rs:133-146`). Its
  `dims` refuses more than 2^32 elements (`init.rs:123-129`). 65536 × 65536 is exactly 2^32, so the
  shape passes and `collect` (`init.rs:142-144`) asks for 16 GiB. Whether that aborts depends on
  the host's RAM, swap and overcommit, so it is no repeatable test, and the review marked the abort
  inferred.

What is repeatable on a 64-bit host is a count under both checks whose byte size no allocator can
grant. An input of `Tensor<f32,[2305843009213693952, 2]>` has 2^62 elements, which passes the
`usize` check of `zeros`, and its 2^64 bytes exceed `isize::MAX`, so `vec![0.0; count]` at
`data.rs:470` fails on every host (by reading, a `capacity overflow` panic; not run). A parameter
cannot reach such a count, because `dims` stops it at 2^32 elements, so its fallible path is tested
at the reservation.

Outcome: a parameter or input whose storage cannot be reserved is a named diagnostic and a non-zero
exit, not a panic or an abort.

## Acceptance

- **oversized-input-named-error**: `train --data <dir>` on a program whose `input` is
  `x: Tensor<f32,[2305843009213693952, 2]>` exits 1 with a named `E-` code and the input's name,
  with no panic message and no abort signal, on a 64-bit host. The story's first test records what
  HEAD does with it; if an earlier stage already refuses the shape, the test names that stage's
  code and this criterion holds as it is.
- **param-reserve-fallible**: parameter storage is reserved with `Vec::try_reserve_exact` before
  the first element is drawn, and a failed reservation maps to the named code. A unit test of that
  reservation with `usize::MAX / 2` elements of `f32` gets the code: the byte size overflows
  `isize::MAX`, so the reservation fails without asking the operating system.
- DESIGN.md § 8.1 names the code.
- `task ci` exits 0.

## Scope

- `crates/modelspec-runtime/src/train/init.rs`
- `crates/modelspec-runtime/src/train/data.rs`
- `DESIGN.md` (§ 8.1)

## Out of scope

- Device-memory exhaustion on Vulkan, which `vkAllocateMemory` already reports.
- Bounds on input files, which story:input-readers-bound-before-allocating holds.
- Lowering the 2^32-element limit of `dims`.
