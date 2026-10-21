---
format: aep.planning-md/3
id: review-result:adversary-m0-cpu-interpreter-pass-1
kind: review-result
status: active
title: Adversary pass 1 on m0-cpu-interpreter
relations:
- reviews: story:m0-cpu-interpreter
revision: 1
---
```
unit: story:m0-cpu-interpreter, uncommitted working tree of ~/.local/state/worktree/trees/b10x/modelspec/modelspec-impl-m0-cpu-interpreter (base e7698c2)
verdict: CONFIRMED (warning); nothing at blocker level, the acceptance statement holds
cases: executed 229→235, red 2
origin: introduced 4 / pre-existing 1 / undecided 0
wrote-outside-worktree: 4 paths (part 6)
needs-coordinator: none. I took no worktree lease; the coordinator's lease covered the tree.
```

## 1. `git --no-pager diff --stat`

```
 DESIGN.md                                       |   3 +-
 crates/modelspec-runtime/src/cpu/interpreter.rs | 332 +++++++++++++++++++++++-
 crates/modelspec-runtime/src/cpu/mod.rs         |   4 +
 crates/modelspec-runtime/src/cpu/tensor.rs      | 148 ++++++++++-
 4 files changed, 484 insertions(+), 3 deletions(-)
```

All four tracked paths are the implementor's and are unchanged by me. The only file I wrote in the tree is the untracked `crates/modelspec-runtime/tests/cpu_adversary.rs`. I formatted only that file, with `rustfmt --edition 2024`.

## 2. Cases added, each run alone before the suite

| Case | Asserts | Now | Alone |
|---|---|---|---|
| `a_value_too_large_for_host_memory_is_a_runtime_error_not_a_panic` | a verified `broadcast_scalar` to `[2^61]` (2^63 bytes, accepted by `verify`) returns `Err`, not a panic | red | exit 101 |
| `mean_divides_by_the_element_count_itself` | f32 mean of `[1, 0 × 2^24]` equals `1/(2^24+1)` rounded, i.e. `2^-24 - 2^-48` | red | exit 101 |
| `mean_divides_and_does_not_multiply_by_a_reciprocal` | `mean([12582914,0,0]) = 4194304.5` in f32 | green | exit 0 |
| `mse_divides_and_does_not_multiply_by_a_reciprocal` | `mse([3548,1,0],0) = 4196101.5` in f32 | green | exit 0 |
| `mse_rounds_each_square_before_adding_it_in_f32` | unfused squares in mse: `0.5+2^-13` in f32; `0.5+2^-13+2^-25+2^-26` in f64 | green | exit 0 |
| `mse_rounds_each_square_before_adding_it_in_f64` | `0.5+2^-28` in f64 | green | exit 0 |

The four green cases exist to kill mutants (part 5). Red output, verbatim:

```
thread '...' panicked at .../library/alloc/src/raw_vec/mod.rs:28:5:
capacity overflow
thread '...' panicked at crates/modelspec-runtime/tests/cpu_adversary.rs:78:13:
the interpreter panicked instead of returning a RuntimeError: capacity overflow
```
```
assertion `left == right` failed: got 5.9604645e-8, expected 5.960464e-8 = 1 / (2^24 + 1) rounded to f32
  left: 864026624
 right: 864026623
```

## 3. Suite

- **`task ci`**: `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci` exited 201. build, `fmt --check` and clippy passed. The test step stopped at the first red binary: `cpu_adversary: FAILED. 4 passed; 2 failed`.
- **Without fail-fast**: the same environment with `cargo test --workspace --no-fail-fast` exited 101. Tally: 233 passed, 2 failed, 235 total. `cpu_golden` passed 78/78; `cpu_adversary` passed 4 and failed 2.

## 4. Findings

| file:line | Verdict | Origin | Measured | What reaches it |
|---|---|---|---|---|
| `crates/modelspec-runtime/src/cpu/interpreter.rs:257` | CONFIRMED | introduced | Too-large value: `vec![..; n]` panics with "capacity overflow". `RuntimeError` has no allocation variant, although DESIGN § 2.2 says "allocations over backend limits are hard errors" and § 8.1 lists "runtime allocation failure". `cpu_adversary.rs:62`, exit 101. | Measured only on a program built with the builder. Inferred, not run: the same unchecked allocation at `:298` (matmul) is reachable from 8 MB inputs (`[2^20,1] x [1,2^20]` gives a 4 TB output). This host has `vm.overcommit_memory = 1`, so that would end in the OOM killer, not a panic, and I did not run it. Fix: check the size with `try_reserve_exact` and return a new `RuntimeError` variant. |
| `DESIGN.md:111` | INFEASIBLE | introduced | "`mean` is that sum divided by the element count": the code divides by the count rounded to f32 (`tensor.rs:46`). For 2^24+1 elements that gives `2^-24`, not `2^-24 - 2^-48`. `cpu_adversary.rs:97`, exit 101. Same for mse at `interpreter.rs:243`. | Nothing found in M0 means a tensor of more than 2^24 elements. Fix is one clause in the DESIGN sentence ("the element count rounded to the element type"), or change the code. |
| `crates/modelspec-runtime/tests/cpu_golden.rs:397` | CONFIRMED | introduced | A fused multiply-add on mse's squares survives `cpu_golden` (78/78 green). My two `mse_rounds_each_square_*` cases kill it. | The no-FMA rule in § 2.2 covers "every product"; only matmul's product was pinned. |
| `crates/modelspec-runtime/tests/cpu_golden.rs:367` | CONFIRMED | introduced | `sum * (1/count)` in place of `sum / count` survives `cpu_golden` for both mean and mse (78/78 green each). My two `*_reciprocal` cases kill them. | "Divided by" in § 2.2 is the oracle a GPU kernel will be compared against (gate 5). |
| `crates/modelspec-runtime/src/cpu/mod.rs:1` | CONFIRMED | pre-existing | The module doc says "Interpreter of tensor **and training IR**". The story says training IR is "never interpreter ops". | The same text is at `e7698c2:crates/modelspec-runtime/src/cpu/interpreter.rs:1`; the unit moved it here. |

## 5. Attacked and could not break

| Attack | Result |
|---|---|
| FMA mutant in matmul (the implementor's claim that it is pinned) | `in_f32::matmul_rounds_each_product_before_adding_it` goes red. Claim holds. |
| Sum starting from the first term (sign of zero) | Holds for sum and matmul; mse squares cannot be -0. |
| Row-major and ascending-k order | Column-major, reverse and pairwise orders are all separated by existing cases. |
| `relu` and `relu_grad`: `>` vs `>=`, `-0` | Caught by existing cases. |
| Scalar-vs-tensor operand order for sub | Caught. |
| Verifier runs first | Moving or removing it fails `the_verifier_runs_before_anything_else`. |
| Shape checks: rank, prefix shape, never-read parameter, undeclared parameter | Each variant has its own red case. |
| Panic paths after verify (operands, outputs, symbolic dimensions, zero dimensions) | None found; the verifier excludes them. |
| `relu(NaN) = 0` | DESIGN is silent on NaN, so this is not a finding. |

## 6. Paths written outside the worktree

1. `~/.cache/modelspec-wave-m0/m0-cpu-interpreter/scratch/adversary/`: a source copy with mutants switched by the `MUT` environment variable. It is 640K, plus its own fresh `target/` of 144M. No target was copied.
2. `~/.cache/modelspec-wave-m0/m0-cpu-interpreter/scratch/adversary-mutants.sh`
3. `~/.cache/modelspec-wave-m0/m0-cpu-interpreter/scratch/adversary-ci.log`
4. `~/.cache/modelspec-wave-m0/m0-cpu-interpreter/scratch/adversary-test-nff.log`

## 7.

```findings
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 257
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: a verified program whose value exceeds host memory panics with capacity overflow instead of returning a runtime allocation error that DESIGN 2.2 and 8.1 require; matmul at :298 has the same unchecked allocation
- file: DESIGN.md
  line: 111
  category: contract-drift
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: mean and mse divide by the element count rounded to f32, not by the element count the policy sentence names, which differs above 2^24 elements
- file: crates/modelspec-runtime/tests/cpu_golden.rs
  line: 397
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: a fused multiply-add on the squares of mse survives cpu_golden; only matmul's no-FMA rule was pinned
- file: crates/modelspec-runtime/tests/cpu_golden.rs
  line: 367
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: multiplying by the rounded reciprocal of the count instead of dividing survives cpu_golden for both mean and mse
- file: crates/modelspec-runtime/src/cpu/mod.rs
  line: 1
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: the module doc claims the CPU backend interprets training IR, which the story rules out
```
