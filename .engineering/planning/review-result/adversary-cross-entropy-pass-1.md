---
format: aep.planning-md/3
id: review-result:adversary-cross-entropy-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-mlm-cross-entropy
relations:
- reviews: story:bert-mlm-cross-entropy
revision: 1
---
```
unit: story:bert-mlm-cross-entropy (impl/bert-mlm-cross-entropy, 0d5d44d over ccdfbc8)
verdict: findings 4 (blocker 0, warning 1, note 3)
cases: executed 22→43, red 5
origin: unit (all 4)
wrote-outside-worktree: <wave-root>/cross-entropy/adv1-scratch only (probe .mspec files, scripts, mutant copy; its target/ removed with cargo clean, 547.6 MiB)
needs-coordinator: F1, fixing it is a D43 choice: either reword DESIGN.md § 4 to "a finite row loss", or skip weight-0 rows (which changes the NaN behaviour a Vulkan kernel must reproduce)
```

| # | Hypothesis | Probe | Observation | Verdict |
|---|---|---|---|---|
| 1 | D43 order, bit for bit | Exp terms `[0.7u, 0.7u, 1, 0.7u]` and weights `[0.7u, 0.7u, 1]` (separate ascending from descending); 200 random draws; f32 and f64; loss and every gradient element | All bits equal | holds |
| 1b | `mul_add` from the second product | Mutant d | Same output as the first-product form: weights and losses are never negative, so `fma(a,b,+0)` equals `a*b` | not observable |
| 2 | Gradient order, `sum(w)` order | Mutants c2 and i | Both killed by the unit's bit test | holds |
| 3 | `-0.0` weights, weight-0 rows, weight sum overflow | 4 cases | `-0` sum gives `+0` loss and gradient; a label out of range in a weight-0 row is refused; an `inf` logit gives NaN; weights `[3e38, 3e38]` give a NaN loss and a zero gradient | F2 |
| 3b | Weight-0 row, finite logits | Logits `[3e38, -3e38, 0]` | Loss is NaN, against DESIGN.md:283 | F1 |
| 4 | Degenerate `V` and logits | `V=1`; `-inf` at the label; a row of all `-inf`; a NaN logit | 0; `inf` with gradient -1 at the label; NaN; NaN; no panic. DESIGN does not say what happens | holds |
| 5 | Checker | 17 probes | E-TYPE-002 comes first for a named argument, then an E-TYPE-005 cascade (`softmax` does the same, so it predates this unit). E-TYPE-004 names `labels`, E-TYPE-018 refuses the third use, E-TYPE-021 is reported at `s` inside the function | N1 |
| 6 | Tied decoder | Tied table gradient compared bit for bit with the untied `enc + dec` gradients, f32 and f64 | Equal; both terms nonzero; one `Add` in the IR | holds |
| 7 | Mutants | 16 planted | 12 killed, 2 equivalent, 2 survive the unit suite | N2 |
| 8 | Docs and spec | `docs-check`, `ui-types-check`, `ess specify validate` | Exit 0, 0, 0; CHANGELOG and DESIGN § 4 match the code | holds |

**F1, warning.** DESIGN.md:283 says a weight-0 row's finite logits "leave the loss's bits unchanged". `max - logit` overflows to `inf`, and `0 * inf` is NaN (story scenario 4 says the same as DESIGN):
```
adv_f1_finite_logits_of_a_zero_weight_row_leave_the_loss_unchanged
assertion `left == right` failed: before 3.2413113, after NaN
```

**F2, note.** The numerator `sum w*loss` is not normalized, so extreme weights that pass the check break the loss. MLM weights are 0 or 1, so a real run does not hit this:
```
adv_f2_weights_whose_sum_overflows_...: loss NaN, f64 2.1269280110429727; gradient [-0.0, 0.0, -0.0, 0.0]
adv_f2_one_large_weight_gives_the_row_loss: loss inf, f64 2.1269280110429727
adv_f2_one_subnormal_weight_gives_the_row_loss: loss 2, f64 2.1269280110429727
```

**N1, note.** With logits `[5,6]`, labels and weights both have length 6, but the E-TYPE-003 message blames `labels`:
```
E-TYPE-003: `cross_entropy` operand shapes differ ... `labels` has type Tensor<i32, [6]>
```

**N2, note.** Two mutants survive `gradients_cross_entropy.rs`: the unit's tests never use an exact 0 weight together with a non-finite logit or a negative `g`. My file now kills both.

| Mutant | Unit suite | My file |
|---|---|---|
| a: max taken in descending order | survives (equivalent) | survives |
| b: exp sum descending | killed by `..._bit_for_bit` | killed |
| c: weight sum descending (forward) | killed by `..._bit_for_bit` | killed |
| c2: weight sum descending (gradient only) | killed by `..._bit_for_bit` | killed |
| d: `mul_add` from the first product | survives (equivalent) | survives |
| e: gradient without `/ sum(w)` | killed by FD w, FD tied, bits | killed |
| f: onehot at `label + 1` | killed by FD w, FD tied, bits | killed |
| g: weight checked before label | killed by `refusals_name_the_first_bad_row` | — |
| h: weight-0 rows skipped (forward) | **survives** | killed by `adv_zero_weight_row_with_an_infinite_logit_makes_the_loss_nan` |
| i: `g * (w / sum)` | killed by `..._bit_for_bit` | killed |
| k: no zero-sum branch (forward) | killed by `refusals_do_not_include_all_weights_0` | killed |
| l: no zero-sum branch (gradient) | killed by `refusals_do_not_include_all_weights_0` | killed |
| n: label check `>` instead of `>=` | killed by 2 refusal tests | killed |
| o: no finite-weight check | killed by `refusals_of_negative_and_non_finite...` | — |
| p: `target - p` | killed by FD w, FD tied, bits | killed |
| q: gradient skips weight-0 rows | **survives** | killed by `adv_random_draws_...` (sign of zero) |

`git status --short`:
```
?? crates/modelspec-cli/tests/adversary_cross_entropy_cli.rs
?? crates/modelspec-runtime/tests/adversary_cross_entropy_numerics.rs
```
Both files pass rustfmt and clippy with `-D warnings`. The lease has ended.

```findings
- file: DESIGN.md
  line: 283
  category: correctness
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "F1: finite logits such as 3e38/-3e38 in a weight-0 row give an infinite f32 row loss, and fma(0, inf, s) makes the loss NaN; the claim that such logits leave the loss bits unchanged is false (crates/modelspec-runtime/src/cpu/interpreter.rs:996; red case adv_f1_finite_logits_of_a_zero_weight_row_leave_the_loss_unchanged)"
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 996
  category: numerics
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "F2: the numerator sum w*loss is not normalized: weights [3e38, 3e38] give a NaN loss with a zero gradient, a weight of 3e38 gives an infinite loss, and a subnormal weight quantizes the loss to 2 against 2.127; red cases adv_f2_*"
- file: crates/modelspec-core/src/types.rs
  line: 1110
  category: diagnostics
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "N1: for logits [5,6] against labels and weights of length 6, the E-TYPE-003 message says '`labels` has type Tensor<i32, [6]>' when the logits have the wrong row count; red case adv_n1_wrong_row_count_of_logits_is_not_blamed_on_labels"
- file: crates/modelspec-runtime/tests/gradients_cross_entropy.rs
  line: 1
  category: test-gap
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: "N2: mutants h (weight-0 rows skipped in the loss) and q (weight-0 rows skipped in the gradient) survive the unit suite, which never uses an exact 0 weight with a non-finite logit or a negative g; adversary_cross_entropy_numerics.rs kills both"
```
