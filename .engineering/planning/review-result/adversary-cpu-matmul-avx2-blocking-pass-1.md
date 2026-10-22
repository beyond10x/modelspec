---
format: aep.planning-md/3
id: review-result:adversary-cpu-matmul-avx2-blocking-pass-1
kind: review-result
status: active
title: Adversary pass 1 on cpu-matmul-avx2-blocking
relations:
- reviews: story:cpu-matmul-avx2-blocking
revision: 1
---
# Adversary pass 1: story:cpu-matmul-avx2-blocking

Agent `aep:adversary`, 2026-10-22, against the uncommitted tree in `modelspec-impl-avx2` over base
`cfb771b`. Report as returned:

```
unit: story:cpu-matmul-avx2-blocking, the uncommitted tree in modelspec-impl-avx2 against base cfb771b
verdict: NEEDS-CHANGE
cases: executed 350→354, red 2 (release profile only)
origin: introduced 2 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory (part 6)
needs-coordinator: none
```

Cases added in `crates/modelspec-runtime/tests/adversary_avx2_kernel.rs`:

| Case | What it asserts | Release | Debug |
|---|---|---|---|
| `the_public_avx2_matmul_panics_when_the_element_counts_overflow` | `matmul::matmul::<f32>(Kernel::Avx2, &[], RowMajor, &[], 2^32, 2^32, 2^32)` panics, as its `# Panics` doc promises | red | green |
| `the_public_element_avx2_hook_panics_when_the_element_counts_overflow` | the same call through `f64::matmul_avx2` (a public, safe method) panics | red | green |
| `each_kernel_equals_the_0_1_1_loop_on_directed_edge_values_in_{f32,f64}` | both kernels and both left-operand layouts match the 0.1.1 loop bit for bit on m in {1,5,6,7,12,13}, n in {1,3,4,5,7,8,9,16,17,33}, overflow to inf mid-sum at k=1500, subnormals, sums of -0.0 and mixed zeros | green | green |

Red output in release: `process didn't exit successfully: ... (signal: 11, SIGSEGV: invalid memory reference)`.

Mechanism: release builds do not check overflow, so `m * k`, `k * n` and `m * n` wrap to 0; the
asserts at `matmul.rs:106-107` and `:223` pass with empty slices, `resize` at `:112` allocates
nothing, and the AVX2 kernel reads and writes through pointers into zero-length buffers. Undefined
behaviour reachable from safe public code. What reaches it: direct calls to the public API only; the
verifier keeps element counts within u64 and the interpreter does not reach it.

Suite after the cases: `cargo test -p modelspec-runtime -p modelspec-cli` exit 0, 354 passed;
`cargo test --release ... --test adversary_avx2_kernel` exit 101, SIGSEGV; clippy and fmt exit 0;
`clippy --target wasm32-unknown-unknown --all-targets -D warnings` exit 101, `variable does not
need to be mutable` at `tests/matmul_row_order.rs:467`.

Attacked without a break: bit equality on the directed shapes in release and debug; masked tail
loads and stores; `out` aliasing; `in_place_transposes` with two consumers, right-operand use,
output use and transpose of transpose; the NaN relaxation (both-NaN only); wasm32 library lint.
Coverage gap: on an AVX2 host the interpreter cannot be forced onto the portable path, only the
kernel API can.

```findings
- file: crates/modelspec-runtime/src/cpu/matmul.rs
  line: 106
  category: boundary
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: In release builds, dimensions whose products wrap to zero pass the unchecked length asserts, and the safe public AVX2 entry points (matmul::matmul, Element::matmul_avx2) then read and write out of bounds and segfault, where their documented contract is a panic.
- file: crates/modelspec-runtime/tests/matmul_row_order.rs
  line: 467
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: kernels() leaves `let mut` unused on non-x86_64 targets, so clippy -D warnings fails there; CI is x86_64 only.
```
