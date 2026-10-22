---
format: aep.planning-md/3
id: review-result:adversary-cpu-matmul-row-order-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:cpu-matmul-row-order
relations:
- reviews: story:cpu-matmul-row-order
revision: 1
---
```
unit: story:cpu-matmul-row-order at ed41930 (worktree modelspec-impl-cpu-r2, plus one untracked test file)
verdict: INFEASIBLE (nothing against the change; one pre-existing note)
cases: executed 310→313, red 0 (3 new cases, green under the gate's dev profile)
origin: introduced 0 / pre-existing 1 / undecided 0
wrote-outside-worktree: 1 path (scratch-adv1/, 531M target inside)
needs-coordinator: decide whether to keep matmul_row_order_adversary.rs, which goes red under `cargo test --release` (finding 1)
```

The change has the same bits as 0.1.1. Every case I could build gave the same bits from both builds, including NaN payloads and signs, in both dev and release. The suite catches every mutant that changes bits.

**1. Diff.** `git diff --stat` is empty, so nothing tracked changed. The only new file is the untracked `crates/modelspec-runtime/tests/matmul_row_order_adversary.rs`. `git diff --stat -- crates/modelspec-runtime/src/cpu/interpreter.rs` is empty. I edited no implementation file in the worktree: every mutant was planted in a copy in scratch, not in the tree.

**2. New cases** (run alone before the suite, dev profile)

| file::name | now | summary line |
|---|---|---|
| matmul_row_order_adversary.rs::nan_payloads_match_the_0_1_1_loop_bit_for_bit_in_f32 | green | `test result: ok. 3 passed; 0 failed` |
| ::nan_payloads_match_the_0_1_1_loop_bit_for_bit_in_f64 | green | same run |
| ::a_sum_of_two_different_nans_keeps_the_0_1_1_operand_in_f32 | green | same run |

- The first two cover 400 shapes up to [8,6]x[6,17], drawn from a pool of NaN with 4 payloads (one signalling), ±inf, ±0 and ±1.
- Under `--release` all 3 go red, on base ab9ed27 and on head alike. Example: `f32 case 0 [2, 2] x [2, 6] element 1: 0.1.1 0xffc00004, got 0x7fc00003`.

**Differential against the real builds.** I hashed the interpreter's output bits over 4 case sets in each dtype, with a scratch test copied into exported copies of base and head (`diff-*.log`).

| | f32: 4 sets | f64: 4 sets |
|---|---|---|
| base dev = head dev | 4/4 equal | 4/4 equal |
| base release = head release | 4/4 equal | 4/4 equal |
| base dev = base release | only the finite set is equal; the 3 NaN sets differ | same |

**Mutants** (planted in scratch copy `mut/`, suite `cargo test -p modelspec-runtime -p modelspec-cli`, `run_mutants.out`)

| mutant | dev | which tests failed |
|---|---|---|
| m1: back to the i, j, k loop | green | none; it is the 0.1.1 code, so it should pass |
| m2: row accumulator starts from +0 | red, 6 | story's 2 bit-equality tests, cpu_golden first-product (f32 and f64), my 2 NaN tests |
| m3: k==0 / n==0 guard dropped | green | none; can't be reached (below) |
| m4: `x*y + *acc` | red, 5 | story's 2 (also red in release), my 3 |
| m5: `y*x` | red, 4 | story's 2 (also red in release), my 2 |
| m6: k in descending order | red, 6 | story's 2, golden ascending-order test, train::data, my 2 |
| m7: guard checks k==0 only | green | none; can't be reached |

The guard can't be reached because the verifier refuses any zero dimension (`crates/modelspec-core/src/ir/verify.rs:220`). I restored the scratch copy afterwards (`cmp` against head is equal).

**3. Gate** (worktree, run after the cases existed)

| command | exit |
|---|---|
| `cargo test -p modelspec-runtime -p modelspec-cli` | 0 (313 passed; the implementor's run had 310) |
| `cargo clippy -p modelspec-runtime --all-targets -- -D warnings` | 0 |
| `cargo fmt -p modelspec-runtime -- --check` | 1 on my file, 0 after I ran `rustfmt` on it; cases re-run green |

**4. Judgement**

| file:line | verdict / origin | finding |
|---|---|---|
| crates/modelspec-runtime/tests/matmul_row_order.rs:187 | INFEASIBLE / pre-existing | The test keeps the 0.1.1 loop as its reference. Its NaN sign and payload are not stable across builds: the real 0.1.1 interpreter gives different NaN bits in dev and in release. So the reference matches the interpreter only in the profile it was compiled in, and the story's own NaN cases pass in release only because of how its values are drawn. Reaching it takes a sum that mixes two different NaNs; I found nobody who trains on NaN data, so I built that case myself. |

**5. Could not break:** -0 sums, ±inf, overflow, subnormals, f64, m/k/n = 1, and the largest shape [64,784]x[784,64].
- There are no non-contiguous operands: `Transpose` copies its data.
- There is no batched matmul: rank 2 is enforced at `crates/modelspec-core/src/ops.rs:122`.
- The backward matmuls autodiff emits go through the same single call site (`interpreter.rs:244`).

**6. Written outside the worktree:** `~/.cache/modelspec-wave-cpu-perf-1/cpu-matmul-row-order/scratch-adv1/` holds the copies `base/`, `head/` and `mut/`, the 7 mutants in `mutants/`, a `target/` of 531M, the scripts and the logs.

```findings
- file: crates/modelspec-runtime/tests/matmul_row_order.rs
  line: 187
  category: property
  severity: note
  verdict: INFEASIBLE
  origin: pre-existing
  message: The 0.1.1 interpreter's NaN sign and payload differ between dev and release builds (same at ab9ed27 and ed41930), so an in-test reference loop is a bit oracle for NaN only in the profile it was compiled in, and distinct-NaN cases go red under cargo test --release on base and head alike.
```
