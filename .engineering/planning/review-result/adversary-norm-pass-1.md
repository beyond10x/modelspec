---
format: aep.planning-md/3
id: review-result:adversary-norm-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-layer-norm-softmax
relations:
- reviews: story:bert-layer-norm-softmax
revision: 1
---
```
unit: story:bert-layer-norm-softmax (head 47922bf, worktree wt-d6caeaf28057 plus untracked adversary files)
verdict: red
cases: executed 137→159, red 2
origin: introduced
wrote-outside-worktree: <wave-root>/norm/adv1-scratch/ (assigned scratch; tree copy, target cleaned)
needs-coordinator: D36 in .engineering/waves/bert-1.md:59 repeats N1's claim ("the general f32 limit is 2.34 u max|x| / s"); that file is the coordinator's
```

**1. Diff.** `git --no-pager diff --stat` is empty: no tracked file changed. `git status --short` shows only untracked new files. All are tests or fixtures; no implementation, spec, docs or `.engineering/` path:
- `crates/modelspec-runtime/tests/adversary_norm_runtime.rs`
- `crates/modelspec-cli/tests/adversary_norm_cli.rs`
- 9 fixtures `crates/modelspec-cli/tests/fixtures/adversary_norm_*.mspec`

**2. Cases.** Each case was run alone first, before any suite run.

| Attack | Case (file:line) | Now | First run alone |
|---|---|---|---|
| 1 | runtime:204 constant row, `LayerNormGradX` against f64 differences at step 1e-9 (c = 0 and 0.5) | green | green |
| 1 | runtime:281 constant row of an inexact value, H=10 (N2) | red, `#[ignore]` N2 | `f32 layer_norm of a constant row of 0.1: [-0.007450374, …]`; after widening the value list: `3.3: [0.23191817, …]`, `77.7: [-0.99151915, …]`, `5000.3: [-0.999998, …]` (f64 gives 0 within 1e-6) |
| 1 | runtime:320 H=1, H=2, gamma=0 | green | red, but **my oracle was wrong**: two values 13 ulps apart have variance below eps, so the gradient really is about 1e6·(gx−mean gx) (`[69855.67, -69864.46]` against f64 `[69754.48, -69754.48]`). I replaced the oracle with an f32-against-f64 check within 1e-2. Not a finding. |
| 1 | runtime:415 element 1e30: f32 variance overflows, output is exactly beta (N3) | green, pins the behaviour | green |
| 1 | runtime:449 softmax K=1, ±1e30, −inf, −0.0, +inf, all −inf | green; NaN on +inf and on an all −inf row (story is silent, so no finding) | green |
| 1 | runtime:486 softmax underflow, logit 200 and 100 above the rest: survivor's gradient is exactly 0 | green | green |
| 2 | runtime:525 LayerNorm, LayerNormGradX, LayerNormGradGamma, SoftmaxGrad, SumRows bit for bit against a hand-written ascending + `mul_add` loop; the descending reference is asserted to differ | green | green |
| 9 | runtime:633 fused against unfused (variance, b, d) and ascending against descending softmax sum, bit for bit, each pair asserted to differ | green | green |
| 3 | runtime:746 rows 1000+v, H=16: variance within 1e-2 of 1 | green | green |
| 4 | runtime:791 / 801 / 817: softmax read twice (once through layer_norm), one param as both gamma and beta, softmax of a param, all against finite differences | green | green |
| 4 | runtime:833 `evaluate` (slots reused) equals `evaluate_all` bit for bit | green | green |
| 8 | runtime:886 the D36 sweep at H=768 (N1) | green, documents N1 | `H = 768: worst ratio, operator 23.60944374322912, correctly rounded mean 0.5283486506565377` |
| 5 | cli:86 / 115 / 137 / 158 / 168: E-TYPE-003 for a long beta; E-TYPE-004 for gamma [1,3], x rank 1, x rank 3, softmax rank 3 and rank 0; E-TYPE-018 for an i32 gamma; E-TYPE-017 | green | green |
| 5 | cli:97 the diagnostic names the wrong operand (N4) | red, `#[ignore]` N4 | `assertion left != right failed: a long gamma and a long beta read the same: E-TYPE-003: layer_norm operand shapes differ / <file>:7:16 / lhs: Tensor<f32, [8, 3]> / rhs: Tensor<f32, [4]> / …` |
| 5 | cli:178 diagnostics.md names both operators | green | green |
| 6 | cli:251 / 257: ir kernel, compile, train and infer on vulkan all give E-TARGET-011 at the call (7:24 layer_norm, 5:16 softmax), print nothing, never create an instance | green | green |
| 7 | cli:305 printed operand order `LayerNorm x, gamma, beta`, then `SoftmaxGrad Y`, `LayerNormGradX xW, gamma, dZ`, `LayerNormGradGamma`, `SumRows` | green | red: a bug in my parser (effect suffix, map order), fixed. Not a finding. |

**3. Suite run, after the cases existed.**

| Command | Result | Exit |
|---|---|---|
| `b10x-build-slot cargo test -p modelspec-runtime --test gradients_norm --test adversary_norm_runtime` | 9 passed; 13 passed, 1 ignored | 0 |
| `… -p modelspec-cli --test check_norm --test adversary_norm_cli` | 5 passed; 9 passed, 1 ignored | 0 |
| `… -p modelspec-core --lib` | 78 passed | 0 |
| `… -p modelspec-docs` | 36 + 9 passed | 0 |

- The 137 "before" cases are the same selection without my two files.
- `cargo fmt --all` changed only my files.
- `clippy -D warnings` passes on modelspec-runtime and modelspec-cli (exit 0).
- `cargo run -p modelspec-docs -- generate --check` exits 0.

**Mutants (attack 9, scratch copy only).** Script at `adv1-scratch/mutants.sh`, results in `mutants.log` and `mutants2.log`.

| Mutant | Killed by the unit's gradients_norm | Killed by an adversary case |
|---|---|---|
| M1 eps 1e-5, M2 no max subtraction, M3 SoftmaxGrad without rowsum, M4 GradGamma sums G, M13 GradX without the xhat·b term | yes | yes |
| M6 mean summed descending | yes (bits test, :298) | yes |
| M7 single-pass `E[x²]−E[x]²` (attack 3) | yes, only `layer_norm_f32_bits_follow_the_two_pass_order` | yes (runtime:746 and others) |
| M5 forward `/ (1/rstd)`, M15 variance unfused | **no** | yes |
| M12 softmax sum descending | **no** | yes (runtime:633) |
| M8 / M9 SoftmaxGrad d descending / unfused, M10 GradX mean_gx descending, M11 GradGamma unfused, M14 GradX b unfused | **no** | yes |

`check_norm` only asserts that the loss falls, so it cannot kill an order mutant.

**4. Findings** (they cover 47922bf plus my untracked files; origin "introduced": every line is new in daa0fbc...47922bf)

| Id | file:line | What was measured | What reaches it | Verdict |
|---|---|---|---|---|
| N1 | DESIGN.md:269 | It says the f32 mean error is "a property of any evaluation that rounds the row and its mean to f32, every row within 2.34 u max\|x\|/s". That was measured at H=5, and H is not stated. At H=768 (BERT width) the operator's worst ratio is 23.6, while a correctly rounded mean gives 0.53. The excess comes from the ascending f32 sum: the worst case grows like H, the typical case like √H. So 2.34 is fitted, not derived. The comment at gradients_norm.rs:239 repeats the "half an ulp" reasoning. | The tolerance a future rowwise Vulkan kernel will be held to; the epic's H=768 | CONFIRMED, warning |
| N2 | crates/modelspec-runtime/src/cpu/interpreter.rs:801 | A constant row whose f32 sum is inexact normalizes to beta ± gamma (−0.9915 for 77.7). The correct result is beta, which f64 and a correctly rounded mean both give. | Nothing found builds such a row (`ones()` rows sum exactly) | INFEASIBLE, note |
| N3 | interpreter.rs:802 | Above about 2e19 the squared deviation overflows: var = inf, rstd = 0, and the output is silently beta, finite. | Nothing found | INFEASIBLE, note |
| N4 | crates/modelspec-core/src/types.rs:756, :780 | A long gamma and a long beta print byte-identical E-TYPE-003. The E-TYPE-004 message never names the wrong operand, unlike add_rows after D33. | Any user who mis-sizes gamma or beta in `modelspec check` | CONFIRMED, note |
| N5 | crates/modelspec-runtime/tests/gradients_norm.rs:298 | The test says it pins the fused variance and the rstd multiply, but its data passes with M15 and M5. | task ci | CONFIRMED, note |
| N6 | gradients_norm.rs:337 | The test says it pins the ascending softmax sum, but its data passes with M12. | task ci | CONFIRMED, note |
| N7 | gradients_norm.rs:360 | No unit test pins the bits of LayerNormGradX, LayerNormGradGamma or SoftmaxGrad, so M8, M9, M10, M11 and M14 survive, although D32 fixes that order so a kernel can reproduce it. runtime:525 and :633 now kill all of them. | task ci | CONFIRMED, warning |

Fix for N1 (a docs line): state H=5, and that the bound grows with H because the f32 sum is recursive.

**5. Attacked and could not break:**
- E-TARGET-011 fires on all four Vulkan commands. The gradient-op arms in schedule.rs cannot be reached: the forward op is refused first.
- `TensorOperator` has its five variants in both schema projections. `ui/src/generated` has no `TensorOperator` at all; that was already so at the base.
- `operators.md` lists both ops with the gradient column, and DESIGN.md § 4 matches the code.
- Autodiff wiring and slot reuse are correct (attack 4).

**6. Paths written outside the worktree** (all under `<wave-root>/norm/adv1-scratch/`):
- `tree/` (copy of the tree; its `target/` was removed with `cargo clean --target-dir`, 548.8 MiB)
- `interpreter.rs.orig`
- `mutants.sh`, `mutants2.sh`
- `mutants.log`, `mutants2.log`
- `mutants.out`, `mutants2.out`

I ended the lease with `session-end`.

```findings
- file: DESIGN.md
  line: 269
  category: docs
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the 2.34 u max|x|/s mean bound was fitted at H=5 and called a property of any f32 evaluation, but at H=768 the operator reaches 23.6 against 0.53 for a correctly rounded mean, so the error is the ascending sum and grows with H
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 801
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a constant f32 row whose ascending sum is inexact (77.7, 5000.3 at H=10) normalizes to beta plus or minus gamma instead of beta
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 802
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: an element above about 2e19 overflows the f32 squared deviation, rstd becomes 0 and layer_norm silently returns beta
- file: crates/modelspec-core/src/types.rs
  line: 756
  category: docs
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: layer_norm's E-TYPE-003 for a long gamma and a long beta is byte-identical and its E-TYPE-004 never names the offending operand, unlike add_rows
- file: crates/modelspec-runtime/tests/gradients_norm.rs
  line: 298
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the layer_norm bit test still passes with an unfused variance or a division by 1/rstd, so it does not pin what its doc comment claims
- file: crates/modelspec-runtime/tests/gradients_norm.rs
  line: 337
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the softmax bit test still passes with a descending exponential sum
- file: crates/modelspec-runtime/tests/gradients_norm.rs
  line: 360
  category: test-quality
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: no unit test pins the bits of LayerNormGradX, LayerNormGradGamma or SoftmaxGrad, so five order and fusion mutants survive the unit's suite
```
