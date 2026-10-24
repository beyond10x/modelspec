---
format: aep.planning-md/3
id: review-result:adversary-cpu-matmul-output-skips-zero-fill-pass-2
kind: review-result
status: active
title: Adversary pass 2 on cpu-matmul-output-skips-zero-fill
relations:
- reviews: story:cpu-matmul-output-skips-zero-fill
revision: 1
---
unit: story:cpu-matmul-output-skips-zero-fill, uncommitted working tree on impl/cpu-matmul-output-skips-zero-fill (base 93121b3), pass 2
verdict: CONFIRMED (3 notes, no red case)
cases: executed 372→373, red 0
origin: introduced 3 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 path
needs-coordinator: I did not plant the zero-fills in the tree. My charter forbids editing the file under attack even briefly, and the brief forbids copying the tree, so the P1–P4 results below come from reading the code, not from runs.

**1. `git --no-pager diff --stat`**
```
 crates/modelspec-runtime/src/cpu/matmul.rs | 168 ++++++++++++++++++++++++++---
 1 file changed, 152 insertions(+), 16 deletions(-)
```
The only change to `matmul.rs` is the implementor's. My one new file is untracked, so the stat does not show it: `crates/modelspec-runtime/tests/adversary_matmul_output_zero_fills.rs`. It is a test file and nothing outside the tests changed.

**2. Case added**

`tests/adversary_matmul_output_zero_fills.rs` holds one test in its own binary. It covers the side the existing guard never checks: does the counter go up when a fill really happens?
- A type with no AVX2 code (`Plain(f64)`) under `Kernel::Avx2`: the counter goes up exactly once per call when `k > 0` and `m*n > 0`, and the call returns `None`.
- The same type with `k = 0`, or with `m*n = 0`: the counter goes up 0 times.
- f32 and f64 under Avx2, and all three types under Portable: 0 times. The output is `Some(m*n)`.
- Both left-operand layouts, 5 shapes with `k > 0`, 3 with `k = 0`, 2 with `m*n = 0`.

It passes now. The run of that case alone gave `test result: ok. 1 passed; 0 failed`. From reading the code, it fails on these mutants, which the implementor's test misses:
- the increment at `matmul.rs:154` deleted;
- the increment moved above the `k == 0` return at `:136`;
- `output_zero_fills()` returning a constant.

**3. Suite run (after the case existed)**

`b10x-build-slot cargo test -p modelspec-runtime --features vulkan` gave EXIT=0: 373 passed, 0 failed, 50 ignored. The full log is in scratch. 372 is the same run with my binary's 1 test subtracted; I did not do a second run to get it.

**4. Findings**
- **Implementor's test only checks for absence** (`tests/matmul_output_zero_fills.rs:48`). It asserts that the count is 0 and nothing else, so it would still pass if the increment or the reader were dead. My case closes that gap and the implementor has nothing to change. Verdict: CONFIRMED. Origin: introduced.
- **A bare fill on the f32/f64 AVX2 path goes unnoticed** (`matmul.rs:180`, plant P2: `out.resize(start+mn, ZERO); out.truncate(start)` before the `reserve` in `avx2_uninit`). No test goes red, because the counter is not touched and f32 has no clone hook. Only the story's profile check (no-memset-in-profile) catches it. This is the limit the implementor already stated; it cannot be tested through the public API. Verdict: INFEASIBLE. Origin: introduced.
- **The new public function's doc is inaccurate** (`matmul.rs:28-32`). `pub fn output_zero_fills` is not marked `#[doc(hidden)]`, unlike `Element::matmul_avx2`. Its doc says the fills are counted "before a kernel wrote over them", but the counted path at `:154-160` also counts calls where the hook then declines and nothing writes over the zeros. Fix: add `#[doc(hidden)]` and reword the doc, or name the function in the CHANGELOG line. What reaches it: no Element type in the repo, since only f32 and f64 implement Element in `src/`. Verdict: CONFIRMED. Origin: introduced.

**5. What I attacked and could not break**
- **Every fill-before-write path counts:** there is exactly one, at `:155`, and its increment is at `:154`. The `k = 0` fill at `:138` is the result itself, not a fill before a write.
- **Where each plant goes red (by reading):**
  - P1, deleting `:150-152` so the AVX2 path goes back to fill-then-hook: `matmul_output_zero_fills.rs` goes red.
  - P3, a fill in `portable`: `matmul_output_fill.rs` goes red through its clone counting.
  - P4, re-adding `resize(end)` before the match: every result test fails on output length.
- **Race on the counter:** none. Only my binary and the implementor's read it, each holds one test, and each test binary is its own process.
- **Uninitialised storage after `set_len`:** I found no case. The kernel stores every row block and every column panel, the remainder under its lane mask (`:409-468`). The pass 1 poisoning allocator already covers m and n from 1 to 33, k from 0 to 17, both layouts, and calls through `evaluate`, so I did not repeat it.
- **`k = 0` exclusion:** it gives `+0` for any m and n, and appends correctly when start > 0. The `m*n = 0` path returns the output unchanged, the same as the base.

**6. Paths written outside the worktree**
- `<wave-root>/zerofill/scratch/adversary/pass2-suite.log`

```findings
- file: crates/modelspec-runtime/tests/matmul_output_zero_fills.rs
  line: 48
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the guard asserts only absence and stays green with the increment at matmul.rs:154 deleted; adversary_matmul_output_zero_fills.rs adds the presence side, no implementation change needed
- file: crates/modelspec-runtime/src/cpu/matmul.rs
  line: 180
  category: mutant
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a bare zero-fill planted in avx2_uninit without the increment turns no test red; only the no-memset-in-profile measurement can catch it, as the implementor stated
- file: crates/modelspec-runtime/src/cpu/matmul.rs
  line: 32
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: output_zero_fills is a test-only instrument exposed as public API without doc(hidden), and its doc says it counts fills a kernel wrote over while it also counts calls whose hook declines
```
