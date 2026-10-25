---
format: aep.planning-md/3
id: review-result:adversary-bert-dense-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-dense-gelu
relations:
- reviews: story:bert-dense-gelu
revision: 1
---
Now clippy on both crates (waits for slots).
unit: story:bert-dense-gelu, tree wt-d0676c91fe93 at 66fc876 plus my untracked test files
verdict: red
cases: executed 1275→1292, red 4
origin: introduced
wrote-outside-worktree: <wave-root>/dense/adv1-scratch (tree/, base/, interpreter.orig.rs, mutants.sh, mutants.log, suite.log, rank1_param.mspec); its target/ removed with `cargo clean --target-dir` (588.8 MiB)
needs-coordinator: whether F2 (no `.mspec` can declare a trainable bias) gets its own story

Verdict in one line: the six acceptance scenarios hold. All 4 red cases are `#[ignore]`d with their finding id. One finding needs a change (F1, a doc comment); the other five are notes. No blocker.

**1. Diff.** `git diff --stat` is empty: nothing tracked was touched. `git status` lists only untracked files, all of them test files or fixtures:
- `crates/modelspec-runtime/tests/adversary_dense_gelu.rs`
- `crates/modelspec-cli/tests/adversary_dense_gelu_cli.rs`
- five fixtures `crates/modelspec-cli/tests/fixtures/adversary_dense_{gelu_only,bias_along_rows,square,bias_rank2,rank3_x}.mspec`

`cargo fmt --all` changed only my CLI file. Clippy with `-D warnings` is clean on both crates.

**2. Cases, each run alone, by attack**

| # | Case (file:line) | Result now | Output when first run |
|---|---|---|---|
| 1 | `adversary_gelu_f32_within_1e_6_relative_at_every_1_over_1024_in_minus_3_to_3` (runtime:104) | green | `worst relative error over [-3, 3] step 1/1024: (9.184994720992196e-7, -2.9501953)` |
| 1 | `…_at_every_f32_in_minus_3_to_minus_1` (runtime:127) | **red, ignored F3** | `gelu: (1.0674767815429152e-6, -2.9703214), 7 points over 1e-6` |
| 1 | `…off_grid_and_extreme_points_match_the_f64_form` (157) | green | see note A |
| 1 | `adversary_the_f64_tanh_reference_underflows_to_zero_in_the_left_tail` (214) | green | see note B |
| 2 | `adversary_gelu_grad_is_half_at_0_symmetric_and_finite_in_the_tails` (229) | green | slope at 0 is exactly 0.5; symmetry error at most 1.19e-7; finite at ±10, ±20, ±88, ±1e3, ±1e10, ±1e15 |
| 2 | `adversary_gelu_grad_is_finite_for_every_finite_input` (280) | **red, ignored F4** | `gelu'(1e20) = NaN` |
| 2 | `adversary_gelu_grad_matches_central_differences_at_3_6_10` (291) | green | — |
| 10 | `adversary_gelu_grad_right_tail_keeps_r_from_its_own_exponential` (261) | green on the unit (1.05e-7) | on mutant M8: `(1.6500308523959006e-6, 4.625)` FAILED |
| 3 | `adversary_add_rows_with_a_zero_dimension_is_refused_not_a_panic` (368), `…n_equal_h_adds_b_by_column` (407) | green | — |
| 4 | `adversary_sum_rows_is_the_ascending_left_fold_not_pairwise` (329), `…bits_equal_an_ascending_fold_and_repeat` (336) | green | 257×7 rows, bit-exact and repeatable; N=1 returns G unchanged |
| 5 | `adversary_a_bias_read_twice_through_gelu_passes_the_gradient_check` (435) | green | N=1 and N=6 |
| 3 | CLI: rows-length `b` gives E-TYPE-003 at 5:16; N=H checks clean; `b [1,3]` and `x [2,3,4]` give E-TYPE-004 at 5:16 | green | — |
| 6 | CLI: `ir --stage kernel`, `compile`, `train` and `infer --backend vulkan` | green | `E-TARGET-011: operator add_rows` at 6:21; `operator gelu` at 6:16 on the gelu-only fixture |
| 6/8 | CLI: `compile --backend cpu` writes artifact.json; 60 CPU training steps lower the loss; `infer --backend cpu` exits 0 | green | loss 0.0579 → 0.0211 |
| 8 | `adversary_the_training_fixture_computes_the_bias_gradient` (cli:229) | **red, ignored F2** | `ir --stage training` lists `GeluGrad`, `MatMul` and `Transpose` but no `SumRows`; gradients only for `@Dense.v` and `@Dense.w` |
| 9 | `adversary_the_diagnostics_page_names_add_rows_for_e_type_003_and_004` (cli:245) | **red, ignored F1** | ``E-TYPE-003: | … `add`, `sub`, `mul` or `mse` operands of different shapes, neither of them rank 0. |`` |

- **Note A.** Every point is within `4 + 3·ceil(cond)` ulps of the f64 form rounded to f32. `cond` is the condition number of gelu at that point, about 230 at x=-10. Signs and special values are right: -0 gives -0, NaN gives NaN, +inf gives +inf.
- **Note B.** The unit's f64 tanh reference returns exactly 0 at x=-10, while the true value is about -1.2e-37. So a relative bound against that reference means nothing below about x=-5. Over [-3, 3] the bound is meaningful.

**3. Suite run.** `b10x-build-slot cargo test -p modelspec-core -p modelspec-runtime -p modelspec-cli` gave 1292 passed, 0 failed, 118 ignored, EXIT=0. My two binaries ran 6+2 ignored (CLI) and 11+2 ignored (runtime). The "before" figure 1275 is 1292 minus those 17, taken from the same run, not from a second one.

**Mutants** (run on the scratch copy only; suites were `--lib`, `gradients_dense` and my file):
- Killed by the unit's suite: M1 (`b[n % h]` instead of `b[h]`), M2 (0.044), M3 (`exp(-u)`), M4 (SumRows descending), M5 (`gelu(A)*G`), M6 (SumRows returns row 0 only), M7 (sigmoid-approximation derivative; the f32 check catches it too), M9 (constant loses its low part).
- **Survived: M8**, which computes `r = 1 - s` at `interpreter.rs:768`. My case at runtime:261 kills it.

**4. Findings** (all against 66fc876)
- **F1, NEEDS-CHANGE, warning** (`crates/modelspec-core/src/types.rs:53`, `:56`). The doc comments that generate `diagnostics.md` still say E-TYPE-003 is "`add`, `sub`, `mul` or `mse`…, neither of them rank 0", and the E-TYPE-004 comment does not mention `add_rows`, which now raises both codes. What reaches it: anyone reading the page after a refused `add_rows`. Fix: name `add_rows` in both comments, then run `task docs-generate`.
- **F2, CONFIRMED, note** (`crates/modelspec-cli/tests/fixtures/dense_gelu_train.mspec:12`). `b` is the training input, so no `train` run ever computes `SumRows`. `check_dense.rs:97` only asserts that losses are finite, not that they fall. A trainable bias also cannot be declared at all: `param b: Tensor<f32,[3]> = xavier_uniform()` is refused with E-TYPE-014 (my probe `rank1_param.mspec`), and `add_rows` refuses a `[1,H]` bias. What reaches it: every dense layer written in `.mspec` today.
- **F3, CONFIRMED, note** (`DESIGN.md:259`, `gradients_dense.rs:218`). DESIGN says the exp form stays "under 1e-6", and the unit's test is named "between the pinned points too". In fact 7 f32 values in [-3, -1] exceed 1e-6, the worst 1.067e-6 at -2.9703214. Scenario 4 (7 fixed points) still holds. What reaches it: a later Vulkan GELU story that reuses 1e-6 at every element.
- **F4, INFEASIBLE, note** (`interpreter.rs:770`). The f32 `GeluGrad` returns NaN for finite |x| ≥ ~5e19, because `0 * inf` occurs when `r` rounds to 0 and the slope overflows. What reaches it: nothing found short of an already diverged run.
- **F5, CONFIRMED, note** (`interpreter.rs:768`). Using `r = 1 - s` passes the whole unit suite, although DESIGN § 4 says `r` takes its own exponential so that neither tail cancels. With the mutant the error is 1.65e-6 at x=4.625; the unit gives 1.05e-7. My case at runtime:261 now pins it.
- **F6, CONFIRMED, note** (`types.rs:746`). For a `b` of shape `[1,3]` the message reads "`add_rows` needs rank-1 operands", which wrongly implies `x` must be rank 1 too. For a rank-1 `x` it reads "rank-2 operands", which implies the same of `b`.

Other points checked:
- A rank-3 `x [B,S,H]` refused with E-TYPE-004 is the story's own choice (`x: [N,H]`), not a finding.
- DESIGN § 4 rows (`dX += G; db += sum_rows(G)`, and the GeluGrad formula) match `ops.rs` and the interpreter.

**5. Could not break**
- The 1e-6 bound on the 1/1024 grid.
- -0, NaN and +inf handling; ±10, ±20, ±88 and subnormals within the conditioning bound.
- GeluGrad: exactly 0.5 at 0, symmetric, agrees with central differences at ±3, ±6, ±10.
- SumRows: ascending order, bit-stable, not pairwise.
- Zero dimensions are refused before the interpreter runs.
- N=H does not confuse the axes.
- E-TARGET-011 reach on all four Vulkan commands.
- CPU compile, train and infer.
- Spec: `ess specify validate` is valid, completeness counts are identical to base 854710c, both schema checks pass, ui-types-check and docs-check exit 0.

**6. Paths written outside the worktree.** Everything is under `<wave-root>/dense/adv1-scratch/`: `tree/` (mutant copy, 13M, target cleaned), `base/spec/`, `interpreter.orig.rs`, `mutants.sh`, `mutants.log`, `suite.log`, `rank1_param.mspec`.

```findings
- file: crates/modelspec-core/src/types.rs
  line: 53
  category: docs
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The E-TYPE-003 and E-TYPE-004 doc comments that generate diagnostics.md do not name add_rows, which now raises both codes.
- file: crates/modelspec-cli/tests/fixtures/dense_gelu_train.mspec
  line: 12
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The training fixture feeds b as input so no train run computes SumRows, and no .mspec can declare a trainable bias (rank-1 xavier is E-TYPE-014, add_rows refuses [1,H]).
- file: DESIGN.md
  line: 259
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The exp-form gelu is not under 1e-6 relative everywhere in [-3,3]; 7 f32 points in [-3,-1] exceed it, worst 1.067e-6 at -2.9703214.
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 770
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: The f32 GeluGrad returns NaN (0 * inf) for finite |x| >= ~5e19, where the derivative tends to 0 or 1.
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 768
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: Mutant r = 1 - s survives the unit suite although DESIGN 4 requires r from its own exponential; it is 1.65e-6 off at x=4.625.
- file: crates/modelspec-core/src/types.rs
  line: 746
  category: docs
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The E-TYPE-004 text for add_rows says it needs rank-1 (or rank-2) operands for all operands, misstating the rule for the other operand.
```
