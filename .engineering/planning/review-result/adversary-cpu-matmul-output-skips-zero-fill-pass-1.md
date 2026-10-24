---
format: aep.planning-md/3
id: review-result:adversary-cpu-matmul-output-skips-zero-fill-pass-1
kind: review-result
status: active
title: Adversary pass 1 on cpu-matmul-output-skips-zero-fill
relations:
- reviews: story:cpu-matmul-output-skips-zero-fill
revision: 1
---
unit: story:cpu-matmul-output-skips-zero-fill, uncommitted working tree on base 93121b3
verdict: CONFIRMED (one note-level judgement finding, no red case)
cases: executed 364→371, red 0
origin: introduced 1 / pre-existing 0 / undecided 0
wrote-outside-worktree: 5 paths (part 6)
needs-coordinator: none

I couldn't break this unit. All 7 cases I added pass, and the one finding is a gap in the tests, not a defect.

**1. `git --no-pager diff --stat`**
```
 crates/modelspec-runtime/src/cpu/matmul.rs | 155 ++++++++++++++++++++++++++---
```
That line is the implementor's change, which was already there when I arrived. I did not touch `matmul.rs`. My only file is the new, untracked `crates/modelspec-runtime/tests/adversary_matmul_output_fill.rs`. `git status --short` shows it next to the implementor's `matmul_output_fill.rs`.

**2. Cases added** (`<trees>/modelspec-impl-zerofill/crates/modelspec-runtime/tests/adversary_matmul_output_fill.rs`)

The test binary installs its own memory allocator. It fills every new block, and the grown part of every resized block, with `0xFF` bytes, which read as NaN in both f32 and f64. The inputs are finite, so an output element the kernel never wrote shows up as NaN and fails a bit-for-bit comparison with the reference sum.
- `the_poisoning_allocator_makes_an_unwritten_element_a_nan`: checks that the detector itself works.
- `every_output_element_is_written_for_every_edge_shape_in_f32` / `_in_f64`: every m and n in 1..=33, every k in 0..=17, both layouts, every kernel this CPU has. Each result is compared bit for bit with the ascending-k reference sum. Inputs include -0, subnormals and fractional values.
- `every_output_element_is_written_through_evaluate`: the interpreter's path, where the output Vec comes from `try_reserve_exact`.
- `a_type_without_avx2_code_is_written_by_the_portable_kernel_and_zeros_where_k_is_0`: under `Kernel::Avx2` a wrapper type is declined when k > 0, gets +0s when k = 0, and gets an empty result when m or n is 0.
- `a_zero_dimension_gives_positive_zeros_from_fresh_storage`
- `a_panicking_call_leaves_nothing_behind_and_the_next_call_is_whole`

First run, that file alone: `test result: ok. 7 passed; 0 failed ... finished in 6.94s`. All green; nothing was red.

**3. Suite and gates** (all after my cases existed)
- `b10x-build-slot cargo test -p modelspec-runtime --features vulkan`: EXIT=0, 371 passed, 0 failed, 50 ignored. Excluding my 7 gives 364.
- `cargo test -p modelspec-cli`: 298 passed, 0 failed. This includes the golden and CLI tests that check per-step loss is unchanged.
- AddressSanitizer: nightly was already installed, so I ran `RUSTFLAGS=-Zsanitizer=address cargo +nightly test --target x86_64-unknown-linux-gnu` on my file, `matmul_output_fill`, `matmul_row_order`, `adversary_avx2_kernel` and `adversary_avx2_counts`. All ok, and the binary contains 24 `__asan_report` symbols, so the sanitizer was really active. MemorySanitizer, the right tool for reads of unwritten memory, can't run without the `rust-src` component, which isn't installed; I didn't install it.
- clippy `-D warnings` (with `--features vulkan`): clean. `cargo fmt --check`: clean, after I ran rustfmt on my own file only.
- Profile check for the story's no-memset criterion: `perf record --call-graph dwarf` on `modelspec train --backend cpu examples/mnist_mlp.mspec` (release build, 1,800 steps, 2,837 samples).
  - `avx2::f32x8::run` takes 81.56%, so the AVX2 kernel is what runs for f32.
  - Of the 44 samples spent in libc, none has a matmul frame in its call stack. The hottest libc addresses are called from `interpreter::outputs`.
  - The criterion holds.

**4. Judgement finding**
- `crates/modelspec-runtime/src/cpu/matmul.rs:138`, a mutant the tests miss. Suppose the branch that calls `avx2_uninit` were deleted. f32 and f64 would then go back to `out.resize(end, T::ZERO)` and `T::matmul_avx2`: the zero-fill is back, and the output values are identical.
  - Nothing in the suite would fail. The implementor's counting test only covers the portable kernel and a type with no AVX2 code, and no test that compares output values can tell the difference.
  - What reaches that path: `interpreter.rs:385` uses `Kernel::detected()`, which is AVX2 for every f32 run on this host. So the code path that actually matters in production has no regression guard; only a profile shows it.
  - This comes from reading the code. I didn't run the mutant because I may not edit implementation files. Severity: note.

**5. Attacked and could not break**
- Unwritten elements for any m, n in 1..=33 and k in 0..=17 (6-row blocks, 8- and 4-lane panels, masked tails): all written.
- Interpreter output with capacity reserved exactly: correct.
- A Vec that already has length or capacity before the call: only reachable through crate-internal `matmul_into`. The public entry points and the interpreter always pass an empty Vec, and the reserve/set_len arithmetic is correct when reading the code.
- A panic leaving unwritten elements visible: every panic comes from an assert before any write or `set_len`.
- Aliasing between operands and output: the borrow checker rules it out for every caller.
- Bit-identity in f32 and f64 for both layouts, including -0 and subnormals, against the ascending-k reference that the base was already tested equal to (`matmul_row_order`).
- k = 0 still gives +0 values. m = 0 or n = 0 gives an empty result, as at the base.
- Types other than f32/f64 under `Kernel::Avx2`: same behaviour as the base.

**6. Paths written outside the worktree**
- `<wave-root>/zerofill/scratch/adversary/perf-after.data`
- `<wave-root>/zerofill/scratch/adversary/train-after.log`
- `<wave-root>/zerofill/scratch/adversary/after.script`
- `<wave-root>/zerofill/scratch/adversary/suite.log`
- `<wave-root>/zerofill/scratch/adversary/` (the directory itself)

Inside the worktree I also created the build directory `target/x86_64-unknown-linux-gnu` (300M, the AddressSanitizer build). I didn't delete it. The session lease `adv-zerofill` is released.

```findings
- file: crates/modelspec-runtime/src/cpu/matmul.rs
  line: 138
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: no test fails if the f32/f64 AVX2 path goes back to zero-fill-then-write, because the outputs are identical; the production path's no-memset property is checked only by a profile, not by the suite
```
