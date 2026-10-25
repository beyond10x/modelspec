---
format: aep.planning-md/3
id: review-result:adversary-attention-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-attention
relations:
- reviews: story:bert-attention
revision: 1
---
```
unit: story:bert-attention, commit 0669da8 on impl/bert-attention (worktree wt-3d13ed30667c, plus 2 untracked adversary test files)
verdict: NEEDS-CHANGE (red: 4 cases, 1 test-quality finding)
cases: executed 2127→2192, red 4
origin: introduced 5 / pre-existing 0 / undecided 0
wrote-outside-worktree: 7 paths (part 6)
needs-coordinator: should F2 (a user program that reaches E-INTERNAL-002) hold the unit or become its own story
```

**1. Diff.** `git --no-pager diff --stat` prints nothing, because both files are untracked. `git status --short` shows only these two, both test files:
```
?? crates/modelspec-cli/tests/adversary_attention_cli.rs
?? crates/modelspec-runtime/tests/adversary_attention_numerics.rs
```
I touched no implementation file. The cases write their temporary sources under the worktree's `target/tmp/`.

**2. My cases.** Each red case was run alone first. The four red ones are now `#[ignore]`d, with the finding id in the reason.

| Attack | Test | Now | Output when first run |
|---|---|---|---|
| 1 | `a1_design_claim_same_bits_holds_for_negative_zero_v` | red, ignored F1 | `6 of 6 elements differ: [0] -0e0 (0x80000000) vs 0e0 (0x00000000) …` |
| 6 | `a6_key_mask_reading_a_parameter_through_a_function_argument` | red, ignored F2 | `E-INTERNAL-002: compiler bug: autodiff refused verified Tensor IR … parameter #1 is read by the loss but has no gradient` (`check` exits 0) |
| 6 | `a6_key_mask_from_a_parameter_free_model_function_checks` | red, ignored F3 | `E-TYPE-021: \`key_mask\` of \`attention\` reads a model parameter … <file>:17:53` |
| 6 | `a6_named_argument_before_the_operands_is_refused` | red, ignored F4 | `left: Some(0) right: Some(1)` |
| 1–5, 7, 9 | 25 numerics cases, other than F1 | green | — |
| 6–10 | 14 CLI cases, other than F2 to F4 | green | — |

Two of my first-run failures were my own test errors, not findings:
- `a5`: I picked two masks that both leave a single key unmasked, so `dK` is all zero either way. I changed the masks.
- The `relu` case: the base commit already reports the `2` in `relu(x, heads = 2)` as E-TYPE-005. I added that to the expectation.

**3. Suite run** (after my cases existed, with `MODELSPEC_VULKAN_LIBRARY` pointing at a file that doesn't exist, no `--ignored`):
- `b10x-build-slot cargo test --workspace --no-fail-fast`: 1581 executed, STATUS1=0.
- `b10x-build-slot cargo test -p modelspec-runtime --features vulkan --no-fail-fast`: 611 executed, STATUS2=0.
- Total 2192. The `before` figure is the same run minus my binaries (15 + 25 + 25 cases). The implementor's figure of 1,376 doesn't match this; I don't know why.
- `cargo fmt --all --check` exits 0; `clippy -D warnings` is clean on modelspec-runtime and modelspec-cli.

**Mutants** (scratch copy of `interpreter.rs`; 8 of 15 survive the unit's `gradients_attention`, and my cases catch all 15):

| Mutant | Unit suite | Caught by (mine) |
|---|---|---|
| sqrt(d) scale, max over all keys, dS without rowsum, dK written at masked keys, dV from P, v head off by one, sequence `n % B` | caught | caught |
| **scale uses H instead of H/A** | survives | a3 ×4, a4 ×2 |
| **softmax sum descending** | survives | a3 ×4 |
| **output key sum descending** | survives | a1, a3 forward |
| **sqrt(1/d) scale** | survives | a3 ×4 |
| **scale computed in f64** | survives | a3 ×4 |
| **scale folded into q** | survives | a3 ×4, a1 |
| **rowsum(P·dP) descending** | survives | a3 dQ, dK |
| **output sum without fusion** | survives | a1, a3 forward |

**4. Findings** (all against 0669da8):
- **F1** `DESIGN.md:279`, docs, CONFIRMED. The claim that skipping a masked key gives "the same bits for every finite k and v" is false when the running sum is `-0.0`: skipping keeps `-0.0`, while the D37 sum with weight `0.0` ends at `+0.0`. The values compare equal. Only a later Vulkan kernel that must reproduce the CPU bits would notice. The code's choice to skip is fine and matches story scenario 3.
- **F2** `crates/modelspec-core/src/types.rs:892`, correctness, NEEDS-CHANGE. `reads_param` treats an `Arg` as reading no parameter. So `mse(model.forward(x, model.mask(m)), y)`, where `mask` reads `p`, passes `check` and then `ir --stage training` and `train` fail with E-INTERNAL-002. DESIGN.md:122 says this case is E-TYPE-021 ("through a model function call"). Who reaches it: any user who computes the key mask in a model function in the train block. A BERT model fed its mask from data does not.
- **F3** `types.rs:889`, judgement, CONFIRMED. Every `model.f()` counts as reading a parameter, so a key mask from a model function that reads none is refused (my source, 17:53). The doc comment there says this over-refusal is intended.
- **F4** `types.rs:707`, contract drift, CONFIRMED. `attention(heads = 2, x, x, x, m)` checks clean, but DESIGN.md:102 puts named arguments after the operands.
- **F5** `crates/modelspec-runtime/tests/gradients_attention.rs:286`, test quality, NEEDS-CHANGE. The unit's suite pins neither the summation order D37 fixes nor the H/A scale for more than one head; 8 mutants survive it.

**5. Attacked and not broken:**
- Masked rows of `dK` and `dV` are exactly `+0`.
- `inf`/NaN in a masked row stays out of every op; NaN in an unmasked row spreads only within its head and sequence.
- The maximum is over unmasked keys only, as D37 says.
- Gradient edges all match finite differences (S=1, one key masked, A=H, B=1, one unmasked among many, saturation at 1e3, tied maximum, v=0, G=0).
- Forward and all three gradients equal my hand-written reference bit for bit.
- Head columns and sequence rows are sliced correctly.
- Autodiff is right with one value as q, k and v, and with the output read twice.
- Mask check: `-0.0` counts as masked; next float above 1, ±inf, -1, a subnormal and NaN are refused, in sequence-then-key order.
- All four Vulkan commands answer E-TARGET-011 at 7:16 without touching a device.
- Checker codes for heads, shapes and i32 masks are as documented; `docs-check`, `spec-validate` and `schema-check` are green.

Two things contradict the implementor's report: `train --backend cpu` does run attention today when the file has one input, and the mask errors exit 1 there. No training-IR golden exists, as for the earlier operators.

**6. Paths written outside the worktree:** `<wave-root>/attention/adv1-scratch/` holding `tree/` (target cleaned, 538.7 MiB removed), `interpreter.rs.orig`, `mutants.sh`, `mutants.log`, `mutants.out`, `suite.sh`, `suite.log`.

**7.**
```findings
- file: DESIGN.md
  line: 279
  category: docs
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: 'The claim that skipping a masked key adds the same bits as weighting it 0.0 for every finite k and v is false: a -0.0 running sum stays -0.0 when skipped and becomes +0.0 under fma(0, v, -0.0).'
- file: crates/modelspec-core/src/types.rs
  line: 892
  category: correctness
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: A key mask that reads a param through a model function argument passes check and ends in E-INTERNAL-002, where DESIGN.md section 2.2 promises E-TYPE-021.
- file: crates/modelspec-core/src/types.rs
  line: 889
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: E-TYPE-021 refuses a key mask computed by a model function that reads no param, because every model function call counts as reading one.
- file: crates/modelspec-core/src/types.rs
  line: 707
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The checker accepts a named argument before the operands, which DESIGN.md section 2.1 places after them.
- file: crates/modelspec-runtime/tests/gradients_attention.rs
  line: 286
  category: test-quality
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The unit's suite pins neither the D37 summation order nor the H/A scale for more than one head, and 8 of 15 interpreter mutants survive it.
```
