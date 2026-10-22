---
format: aep.planning-md/3
id: review-result:adversary-cpu-matmul-avx2-blocking-pass-2
kind: review-result
status: active
title: Adversary pass 2 on cpu-matmul-avx2-blocking
relations:
- reviews: story:cpu-matmul-avx2-blocking
revision: 1
---
# Adversary pass 2: story:cpu-matmul-avx2-blocking

Agent `aep:adversary`, 2026-10-22, against the corrected uncommitted tree in `modelspec-impl-avx2`
over base `cfb771b`. Report as returned:

```
unit: story:cpu-matmul-avx2-blocking pass 2, the uncommitted tree in modelspec-impl-avx2 against base cfb771b
verdict: nothing found
cases: executed 354→359, red 0
origin: introduced 0 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory (part 6)
needs-coordinator: none
```

Cases added in `crates/modelspec-runtime/tests/adversary_avx2_counts.rs`, all green in release and
debug:

| Case | What it asserts |
|---|---|
| `only_the_result_count_overflows_and_every_kernel_panics` | m=n=2^33, k=0: only `m*n` overflows, and every kernel panics |
| `a_right_operand_count_that_overflows_with_m_zero_panics` | m=0, k=n=2^40, transposed layout: panics |
| `slices_longer_or_shorter_than_their_counts_panic` | a or b one element longer, one shorter, or empty: panics on every kernel |
| `the_element_hook_checks_out_as_well` | `f32::matmul_avx2` with an `out` of 3 or 5 elements for a 2x2 result: panics |
| `a_zero_dimension_still_gives_m_times_n_positive_zeros` | 6 shapes with a zero dimension, both kernels and layouts: m*n elements, every bit pattern +0 |

Gate after the cases: debug `cargo test -p modelspec-runtime -p modelspec-cli` exit 0, 359 passed;
release `matmul_row_order` 13 passed; release adversary files 5 + 4 passed; clippy `-D warnings`
exit 0 on the host and on wasm32; `cargo fmt --check` exit 0.

Attacked without a break: every path into the unsafe kernel goes through `element_counts`
(`matmul`, `matmul_into`, both `avx2::*::matmul`, the default `Element::matmul_avx2`); `run` is a
private `unsafe fn` and the first pointer is formed after the exact-length asserts at
`matmul.rs:243-244`; wrong slice lengths panic; zero dimensions still return m*n +0 values; bits
unchanged.

```findings
[]
```
