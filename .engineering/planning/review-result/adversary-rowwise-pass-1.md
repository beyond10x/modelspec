---
format: aep.planning-md/3
id: review-result:adversary-rowwise-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-vulkan-kernels-rowwise
relations:
- reviews: story:bert-vulkan-kernels-rowwise
revision: 1
---
Final state is right. Now pulling the verbatim red outputs from this log.
unit: story:bert-vulkan-kernels-rowwise (worktree wt-ed2b026afe91, head 7770108)
verdict: findings 5 (blocker 0, warning 5, note 0)
cases: executed 1,122→1,127 (and 14 device cases run with `--ignored`), red 4
origin: introduced, all 5
wrote-outside-worktree: yes, only `<wave-root>/rowwise/adv1-scratch/`
needs-coordinator: yes. (1) I set the build budget to 56/70/80, not 52. The repository footprint grew to 53→72G from other trees while this tree's `target/` stayed about 15G. (2) The brief expects "exit 2" for the dropout refusal, but the code exits 1, the same as every other diagnostic. That is an error in the brief, not a finding.

```
$ git --no-pager diff --stat
 .engineering/waves/bert-1.md | 2 +-        (the coordinator's, untouched)
```
My two files are untracked: `crates/modelspec-runtime/tests/adversary_rowwise.rs` (377 lines, 5 host cases, gated on `vulkan`) and `.../adversary_rowwise_device.rs` (1,224 lines, 14 `#[ignore]` device cases). Both pass rustfmt and `clippy --features vulkan --tests -D warnings`. Without the feature they compile to 0 tests.

## Red cases (verbatim, from `adversary_rowwise_device -- --ignored`)

**ADV-ROWWISE-1** `adv_rowwise_1_layer_norm_of_a_constant_row_is_within_tolerance`
```
4 of 8 constant rows normalize to exactly beta on the CPU
panicked at crates/modelspec-runtime/tests/adversary_rowwise_device.rs:1093:9:
    "LayerNorm [0.1; 3] [0]: gpu -0.007450374 (0xbbf42244) cpu 0.0 (0x00000000)",
    "LayerNorm [0.1; 3] [1]: gpu -0.007450374 (0xbbf42244) cpu 0.0 (0x00000000)",
    "LayerNorm [0.1; 3]: 3 elements disagree in all",
    "LayerNormGradGamma [0.1; 3] [0]: gpu 0.007450374 (0x3bf42244) cpu -0.0 (0x80000000)",
    "LayerNormGradGamma [0.1; 3] [2]: gpu -0.007450374 (0xbbf42244) cpu 0.0 (0x00000000)",
    "LayerNorm [0.3; 7] [0]: gpu 0.0 (0x00000000) cpu 0.029789098 (0x3cf40844)",
    "LayerNorm [0.3; 7] [1]: gpu 0.0 (0x00000000) cpu 0.029789098 (0x3cf40844)",
    "LayerNorm [0.3; 7]: 7 elements disagree in all",
    "LayerNormGradX [0.3; 7] [0]: gpu -3000000.0 (0xca371b00) cpu -2998668.8 (0xca370633)",
    "LayerNormGradX [0.3; 7] [1]: gpu -2000000.0 (0xc9f42400) cpu -1999112.5 (0xc9f40844)",
    "LayerNormGradX [0.3; 7]: 7 elements disagree in all",
    "LayerNormGradGamma [0.3; 7] [0]: gpu -0.0 (0x80000000) cpu -0.08936729 (0xbdb70633)",
    "LayerNormGradGamma [0.3; 7] [1]: gpu -0.0 (0x80000000) cpu -0.059578195 (0xbd740844)",
    "LayerNormGradGamma [0.3; 7]: 6 elements disagree in all",
    "LayerNorm [0.7; 11] [0]: gpu 0.059499044 (0x3d73b545) cpu 0.118371174 (0x3df26c96)",
    "LayerNorm [0.7; 11] [1]: gpu 0.059499044 (0x3d73b545) cpu 0.118371174 (0x3df26c96)",
    "LayerNorm [0.7; 11]: 11 elements disagree in all",
    "LayerNormGradX [0.7; 11] [0]: gpu -4991141.5 (0xca98514b) cpu -4964847.0 (0xca9783de)",
    "LayerNormGradX [0.7; 11] [1]: gpu -3992913.3 (0xca73b545) cpu -3971877.5 (0xca726c96)",
    "LayerNormGradX [0.7; 11]: 11 elements disagree in all",
    "LayerNormGradGamma [0.7; 11] [0]: gpu -0.29749522 (0xbe98514b) cpu -0.5918559 (0xbf1783de)",
    "LayerNormGradGamma [0.7; 11] [1]: gpu -0.23799618 (0xbe73b545) cpu -0.4734847 (0xbef26c96)",
    "LayerNormGradGamma [0.7; 11]: 10 elements disagree in all",
    "LayerNorm [12.34; 13] [0]: gpu 0.69014585 (0x3f30ad66) cpu 0.88565725 (0x3f62ba6f)",
    "LayerNorm [12.34; 13] [1]: gpu 0.69014585 (0x3f30ad66) cpu 0.88565725 (0x3f62ba6f)",
    "LayerNorm [12.34; 13]: 13 elements disagree in all",
    "LayerNormGradX [12.34; 13] [0]: gpu -4342022.0 (0xca84820c) cpu -2786036.8 (0xca2a0bd3)",
    "LayerNormGradX [12.34; 13] [1]: gpu -3618352.0 (0xca5cd8c0) cpu -2321697.3 (0xca0db485)",
    "LayerNormGradX [12.34; 13]: 13 elements disagree in all",
    "LayerNormGradGamma [12.34; 13] [0]: gpu -4.140875 (0xc084820c) cpu -5.3139434 (0xc0aa0bd3)",
    "LayerNormGradGamma [12.34; 13] [1]: gpu -3.4507294 (0xc05cd8c0) cpu -4.428286 (0xc08db485)",
    "LayerNormGradGamma [12.34; 13]: 12 elements disagree in all",
```
The [0.2; 768] row is also off: LNGX worst |g−c| 6.250e-2.

**ADV-ROWWISE-2** `adv_rowwise_2_layer_norm_grad_x_of_a_cancelling_gradient_is_within_tolerance`
```
LNGX [4096, 2]: 5056 of 8192 CPU elements 0, worst 1.719e-4
panicked at crates/modelspec-runtime/tests/adversary_rowwise_device.rs:1165:9:
    "LNGX [4096, 2] [442]: gpu 1.1082743e-5 (0x3739f005) cpu 0.0 (0x00000000)",
    "LNGX [4096, 2] [443]: gpu -1.1082743e-5 (0xb739f005) cpu 0.0 (0x00000000)",
    "LNGX [4096, 2] [932]: gpu 1.4905239e-5 (0x377a1184) cpu 0.0 (0x00000000)",
    "LNGX [4096, 2]: 50 elements disagree in all",
    "LNGX g = y, h 64 spread 0.035 [2]: gpu 1.8517629e-5 (0x379b564e) cpu 6.1725436e-6 (0x36cf1dbf)",
    "LNGX g = y, h 64 spread 0.035 [5]: gpu -1.8517629e-5 (0xb79b564e) cpu -6.1725436e-6 (0xb6cf1dbf)",
    "LNGX g = y, h 64 spread 0.035 [8]: gpu -1.8517629e-5 (0xb79b564e) cpu -6.1725436e-6 (0xb6cf1dbf)",
    "LNGX g = y, h 64 spread 0.035: 128 elements disagree in all",
    "LNGX g = y, h 64 spread 0.01 [256]: gpu 6.764141e-5 (0x388ddab5) cpu -2.2547136e-5 (0xb7bd239b)",
    "LNGX g = y, h 64 spread 0.01 [257]: gpu 0.0 (0x00000000) cpu 2.2547136e-5 (0x37bd239b)",
    "LNGX g = y, h 64 spread 0.01 [258]: gpu 6.764141e-5 (0x388ddab5) cpu -2.2547136e-5 (0xb7bd239b)",
    "LNGX g = y, h 64 spread 0.01: 194 elements disagree in all",
    "LNGX g = y, h 768 spread 0.035 [4]: gpu -6.8783454e-5 (0xb8903fd6) cpu -8.0247366e-5 (0xb8a84a7a)",
    "LNGX g = y, h 768 spread 0.035 [10]: gpu -6.30515e-5 (0xb8843a84) cpu -7.451541e-5 (0xb89c4528)",
    "LNGX g = y, h 768 spread 0.035 [14]: gpu -7.451541e-5 (0xb89c4528) cpu -8.597932e-5 (0xb8b44fcc)",
    "LNGX g = y, h 768 spread 0.035: 1635 elements disagree in all",
```

**ADV-ROWWISE-3** `adv_rowwise_3_bit_claims_hold_for_subnormal_operands`
```
panicked at crates/modelspec-runtime/tests/adversary_rowwise_device.rs:625:9:
    "AddRows [0]: gpu 0.0 (0x00000000) cpu 1.148e-41 (0x00002000)",
    "AddRows [1]: gpu -0.0 (0x80000000) cpu -2.20455e-40 (0x8002668a)",
    "AddRows [2]: gpu 0.0 (0x00000000) cpu 4.29431e-40 (0x0004ad14)",
    "AddRows [3]: gpu -0.0 (0x80000000) cpu -6.38406e-40 (0x8006f39e)",
    "AddRows: 35 elements disagree in all",
    "SumRows [0]: gpu 0.0 (0x00000000) cpu 1.468569e-39 (0x000ffdc6)",
    "SumRows [1]: gpu 0.0 (0x00000000) cpu -1.573057e-39 (0x8011210b)",
    "SumRows [2]: gpu 0.0 (0x00000000) cpu 1.677545e-39 (0x00124450)",
    "SumRows [3]: gpu 0.0 (0x00000000) cpu -1.782033e-39 (0x80136795)",
    "SumRows: 7 elements disagree in all",
    "EmbeddingGrad [0]: gpu 0.0 (0x00000000) cpu 2.937138e-39 (0x001ffb8c)",
    "EmbeddingGrad [1]: gpu 0.0 (0x00000000) cpu -3.146114e-39 (0x80224216)",
    "EmbeddingGrad [2]: gpu 0.0 (0x00000000) cpu 3.35509e-39 (0x002488a0)",
    "EmbeddingGrad [3]: gpu 0.0 (0x00000000) cpu -3.564065e-39 (0x8026cf2a)",
    "EmbeddingGrad: 21 elements disagree in all",
    "Dropout [1]: gpu -0.0 (0x80000000) cpu -2.20455e-40 (0x8002668a)",
    "Dropout [3]: gpu -0.0 (0x80000000) cpu -6.38406e-40 (0x8006f39e)",
    "Dropout [5]: gpu -0.0 (0x80000000) cpu -1.056358e-39 (0x800b80b2)",
    "Dropout [9]: gpu -0.0 (0x80000000) cpu -1.89226e-39 (0x80149ada)",
    "Dropout: 15 elements disagree in all",
    "DropoutGrad [1]: gpu -0.0 (0x80000000) cpu -2.20455e-40 (0x8002668a)",
    "DropoutGrad [3]: gpu -0.0 (0x80000000) cpu -6.38406e-40 (0x8006f39e)",
    "DropoutGrad [5]: gpu -0.0 (0x80000000) cpu -1.056358e-39 (0x800b80b2)",
    "DropoutGrad [9]: gpu -0.0 (0x80000000) cpu -1.89226e-39 (0x80149ada)",
    "DropoutGrad: 15 elements disagree in all",
```
`adv_rowwise_3_bit_claims_hold_against_the_flushed_reference_for_subnormal_results`. The operands are normal and the results subnormal, so the flushed reference equals the CPU's:
```
panicked at crates/modelspec-runtime/tests/adversary_rowwise_device.rs:1222:9:
    "AddRows [0]: gpu 0.0 (0x00000000) cpu 4.408104e-39 (0x00300000)",
    "AddRows [1]: gpu -0.0 (0x80000000) cpu -4.408104e-39 (0x80300000)",
    "AddRows [2]: gpu 0.0 (0x00000000) cpu 2.204052e-39 (0x00180000)",
    "AddRows [3]: gpu -0.0 (0x80000000) cpu -5.51013e-39 (0x803c0000)",
    (SumRows and EmbeddingGrad: the same four lines each)
```

## Suite run, made after the cases existed
- `B10X_BUILD_REPO_BUDGET_G=80 b10x-build-slot cargo test -p modelspec-runtime -p modelspec-cli --features vulkan`: exit 0, 182 suites, 1,127 passed, 0 failed, 156 ignored. The implementor's run was 180 / 1,122 / 142. The difference is my 5 host cases and 14 device cases.
- `--test adversary_rowwise_device -- --ignored`: exit 101, 10 passed, 4 failed (the 4 red cases above).
- `--test vulkan_bert_rowwise -- --ignored`: 4 passed.
- `matmul_untransposed_modules_unchanged`: `every_kernel_signature_of_the_plan_before_the_change_emits_the_words_it_emitted_then ... ok`.
- `cargo build -p modelspec-runtime` without `vulkan`: exit 0.

## Judgement findings
| id | sev | where | finding |
|---|---|---|---|
| ADV-ROWWISE-1 | warning | `spirv/emit/rowwise.rs:408` | The mean comes from `OpFDiv`, which is not correctly rounded (2.5 ulp). On a constant row it lands one ulp off, d becomes ±ulp, and rstd ≈ 1e6 amplifies that: LN off by 0.2, LNGX off by 1.5e6. Fix: correct the quotient with an fma residual (`q=a/b; r=fma(-q,b,a); q+=r/b`), and treat sqrt and the reciprocal the same way. |
| ADV-ROWWISE-2 | warning | `rowwise.rs:586`, DESIGN § 9 gate 5 | LNGX error is rstd error times rstd. With r=\|c\| it fails where the CPU cancels to 0 or to a small value, including g = the row's own LN output, which is the mse(LN(x),0) gradient. Either correct rstd (as in ADV-ROWWISE-1) or define r for per-row kernels as Σ\|terms\|. |
| ADV-ROWWISE-3 | warning | `DESIGN.md:314` ("give the CPU's bits on every device"), `DESIGN.md:4` | The 3090 flushes subnormal operands and results, and § 5.4 itself puts devices without DenormPreserve outside the bit class. The claim fails on the only device. The unit's `Draw` holds no subnormals. |
| ADV-ROWWISE-4 | warning | `tests/vulkan_bert_rowwise.rs:180` | Mutant (b) survives the unit's suite. My green `adv_rowwise_edge_values_agree_with_the_cpu` kills it: `"Softmax specials [12]: gpu NaN (0x7fffffff) cpu 1.0 (0x3f800000)"`. |
| ADV-ROWWISE-5 | warning | `tests/vulkan_bert_rowwise.rs:148` | Mutant d2 (SumRows from +0.0, not from row 0) survives the unit's suite. My new green `adv_rowwise_sum_rows_of_negative_zero_columns_is_negative_zero` kills it: `"SumRows -0.0 n 1 [0]: gpu 0.0 (0x00000000) cpu -0.0 (0x80000000)"`. |

## Hypotheses
| # | probe | observation | verdict |
|---|---|---|---|
| 1 | 9 shapes: N 1/64/65/4096, D 1/2/64/65/1024, V 1/50,000; LN of 1 element; softmax of 1 column; SumRows over 100k rows | all within tolerance; SumRows bit-equal in about 20 ms | holds, except ADV-2 at D=2 |
| 2 | ±inf, NaN, MAX, ±100, 1e20, subnormals, constant rows, out-of-table ids | specials agree; an out-of-table id writes 0.0, as § 5.2 states, but E-DATA-004 makes that path unreachable today | ADV-1, ADV-3 |
| 3 | 10 masks of 1,000,003 elements over 3 sites plus one of 4,194,311 (2D grid); p = 0, 0.999, 1−ulp, MIN_POSITIVE; workgroup 128 in the scratch copy | 14.19M elements bit-equal; mask unchanged at workgroup 128 | holds |
| 4 | EmbeddingGrad ×20 runs, 8,192 ids into one row; LNGG at n = 1…500 around 64-row block edges | identical bits; LNGG worst 3.8e-5, two runs bit-equal | holds |
| 5 | flop/byte counts by hand for 9 ops; chain fusion around row-wise ops | all match; no row-wise op inside a chain | holds |
| 6 | spirv-val on the extended grid; NoContraction on OpFDiv; digests | 520 modules pass; 136 OpFDiv all decorated; digests unchanged | holds |
| 7 | the 8 updated tests; remaining E-TARGET-011 sites | all hold; the cross_entropy-at-compile check was missing and my host case adds it | holds |
| 8 | 200 steps CPU vs vulkan on 3 models, checkpoint at 100, cross-resume both ways | losses agree to about 1e-7; dense_gelu final 0.24798436 / 0.24798438 | holds |
| 9 | mutants (table below) | b and d2 survive the unit's suite | ADV-4, ADV-5 |
| 10 | diagnostics.md, status.json, CHANGELOG | match the code; the test status.json names exists | holds |

## Mutants (scratch copy; base run: unit tests green, my 4 reds red)
| mutant | unit suite | my cases |
|---|---|---|
| a1 first square as fma(d,d,0) (the brief's literal (a)) | survives (equivalent mutant, same bits) | survives |
| a2 squares never fused | killed: `every_rowwise_float_op_is_uncontracted_except_under_fast_math` | (structural only) |
| b Softmax max ignores x[0] | **survives** | killed: edge_values |
| c EmbeddingGrad descending n | killed: embedding_grad_repeated_ids, rowwise_kernels_match | killed |
| d1 SumRows drops row 0 | killed: rowwise_kernels_match, trains_a_matmul_add_rows_gelu | killed |
| d2 SumRows from +0.0 | **survives** | killed: sum_rows_of_negative_zero_columns |
| d3 SumRows from row 1, row 0 added last | killed: rowwise_kernels_match | killed |
| e GeluGrad 2·cubic instead of 3·cubic | killed: rowwise_kernels_match, trains_… | killed |
| f Dropout counter index+1 | killed: dropout_masks_…bit_for_bit | killed |

## Written outside the worktree
Everything under `<wave-root>/rowwise/adv1-scratch/` (20M in all):
- `mut/` is a source copy; its target has been cleaned and is gone.
- `h7/` and `h8/` hold the training runs and checkpoints.
- Other files: `chain.mspec`, `chain.ir`, the `*.orig` copies (rowwise.rs, emit.rs, kernel.rs, schedule.rs), `mutrun.sh`, `diskwatch.sh/.log/.stop`, `mut-*.log`, `suite-*.log`, `device-*.log`.

The disk watcher has exited, no process of mine is still running, and the lease is ended.

```
$ git status --short
 M .engineering/waves/bert-1.md
?? crates/modelspec-runtime/tests/adversary_rowwise.rs
?? crates/modelspec-runtime/tests/adversary_rowwise_device.rs
```

```findings
- file: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
  line: 408
  category: correctness
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "ADV-ROWWISE-1: row_moments divides the mean with OpFDiv (2.5 ulp); on constant rows the one-ulp mean error is amplified by rstd ~1e6, LayerNorm off by up to 0.196, LayerNormGradX by 1.556e6, LNGG by 1.173 against the CPU; correct the quotient with an fma residual and do the same for sqrt and the reciprocal."
- file: crates/modelspec-runtime/src/spirv/emit/rowwise.rs
  line: 586
  category: correctness
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "ADV-ROWWISE-2: LayerNormGradX fails gate 5 at r=|c| where the CPU result cancels (width-2 rows: 50 elements; g = layer_norm output: 128, 194 and 1635 elements); fix rstd accuracy or define r for per-row kernels as the sum of |terms| in DESIGN § 9."
- file: DESIGN.md
  line: 314
  category: docs
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "ADV-ROWWISE-3: 'give the CPU's bits on every device' (and revision 0.46, line 4) is false on the RTX 3090, which flushes subnormal operands and results in AddRows, SumRows, EmbeddingGrad, Dropout and DropoutGrad; § 5.4 already excludes devices without DenormPreserve from the bit class."
- file: crates/modelspec-runtime/tests/vulkan_bert_rowwise.rs
  line: 180
  category: test-coverage
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "ADV-ROWWISE-4: mutant (b), Softmax max that ignores x[0], survives the unit's suite; adv_rowwise_edge_values_agree_with_the_cpu kills it."
- file: crates/modelspec-runtime/tests/vulkan_bert_rowwise.rs
  line: 148
  category: test-coverage
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: "ADV-ROWWISE-5: mutant d2, SumRows starting from +0.0 instead of row 0, survives the unit's suite; adv_rowwise_sum_rows_of_negative_zero_columns_is_negative_zero kills it."
```
