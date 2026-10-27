---
format: aep.planning-md/3
id: review-result:adversary-transformer-block-pass-1
kind: review-result
status: active
title: Adversary review of transformer-block-benchmark at 97a78ac
summary: 'Unit 22 adversary pass 1: 7 findings (0 blocker, 3 warning, 4 note), 5 red cases, D84 fix pass'
relations:
- reviews: story:transformer-block-benchmark
revision: 1
---
# Adversary review of story:transformer-block-benchmark at 97a78ac

unit: story:transformer-block-benchmark (unit 22), unit commit 97a78ac on plan commit 7f7403e
verdict: NEEDS-CHANGE
cases: workspace suites executed 1949 → 1955 (0 failed, 204 ignored); 5 red cases, kept `#[ignore]` with the finding id in their name
origin: introduced 6 / pre-existing 1 / undecided 0

## Cases added

Three new files, left in the tree: `crates/modelspec-runtime/tests/adversary_transformer_block.rs`,
`crates/modelspec-cli/tests/adversary_transformer_block.rs`,
`crates/modelspec-docs/tests/adversary_transformer_block.rs`.

| case | asserts | result |
|---|---|---|
| runtime `adv_tb_1_tokens_per_second_and_gflops_are_over_the_median_of_distinct_runs` | three distinct runs; tokens/s and GFLOP/s from the median | green; kills mutant b |
| runtime `the_closed_form_is_the_product_by_product_recount_at_every_batch` | 339,738,624·B at B = 1, 3, 8, 32, 128, 1000 | green |
| runtime `two_key_mask_inputs_are_both_all_ones`, `the_example_s_key_mask_is_its_only_key_mask_at_every_batch` | the key-mask rule | green |
| runtime `adv_tb_5_…mse_of_2_million_elements…` | CPU `mse` against an f64 sum within the § 9 bound | red: |diff| 1.302e-3 > bound 1.099e-3 |
| runtime `adv_tb_5_…step_0_loss_at_batch_128…` | CPU step-0 loss at B = 128 against the f64 sum | red: |diff| 1.170e-3 > bound 8.494e-4 |
| cli `infer_on_the_step_0_checkpoint_prints_the_step_0_loss`; `--write-batch` onto a file exits 1 | green |
| cli `adv_tb_4_the_written_batch_of_the_example_reads_back_as_a_dense_data_directory` | `--data` on the written batch | red: E-DATA-001, inputs: 2 |
| docs `block_section_figures_are_recomputed_from_the_entry` | green |
| docs `adv_tb_2_…without_tokens_per_step…` | a `transformer-block` row without `tokens_per_step` is refused | red: accepted, drawn in the MLP-sweep layout |
| docs `adv_tb_3_…without_a_batch` | `tokens_per_step` without `batch` is refused | red: accepted |

Device runs the brief allows: `ten_vulkan_steps_agree_with_the_cpu_run` ok in 18.67 s; bench smoke
`--point 32 --point 7 --min-seconds 1` measured (B = 32 median 172,720.8 µs, `tokens_per_s` and
`gflops` recompute from the median); opponent eager smoke at B = 32 from the B = 32 checkpoint
measured (median 4,096.4 µs).

## Findings

| id | severity | file:line | finding |
|---|---|---|---|
| ADV-TB-5 | warning, pre-existing | `crates/modelspec-runtime/src/cpu/interpreter.rs:850` | the CPU `mse` sums all squared differences in one `f32` running total; at B = 128 (2,097,152 elements) it is 1.56e-3 off the f64 value, so the step-0 loss 0.7482758 is outside the run tolerance of PyTorch's 0.74944544 (f64 reference 0.749445488; PyTorch is the accurate side). At B = 8 and B = 32 both are within tolerance. Reproduces at the base 7f7403e. |
| ADV-TB-6 | warning, introduced | `opponents/transformer_block/README.md:69` | the step-0 loss check at the run tolerance (bound 8.48e-4 at B = 8) passes an opponent with the erf GELU (|Δ| 5.8e-5), a GELU coefficient of 0.04 (4.0e-4) and eps 1e-5 (3.6e-6); it catches the attention scale 1/√H (0.0127) and a dropped last layer norm (0.096). The README's "shown to compute the same function" is not shown by that check. |
| ADV-TB-2 | warning, introduced | `crates/modelspec-docs/src/benchmarks.rs:1701` | a `transformer-block` row with `batch` and the `MatmulOnly` count but no `tokens_per_step` is accepted and drawn in the MLP-sweep layout ("each hidden width H", an `H none` row, no tokens-per-second column). |
| ADV-TB-1 | note, introduced | `crates/modelspec-runtime/tests/transformer_block.rs:105` | the median test feeds three equal runs, so a mean-instead-of-median mutant passes all 8 unit cases. |
| ADV-TB-3 | note, introduced | `crates/modelspec-docs/src/benchmarks.rs:260` | the 128·B check on `tokens_per_step` runs only when the row also gives `batch`. |
| ADV-TB-4 | note, introduced | `crates/modelspec-cli/src/train.rs:257` | `--write-batch` claims "the layout of a dense `--data` directory", but `--data` on the example's written batch is refused with E-DATA-001 (the key mask counts as a second input). |
| ADV-TB-7 | note, introduced | `opponents/transformer_block/transformer_block.py:115` | the opponent's `.msckpt` reader loads a payload with byte 200003 flipped and prints loss 1.3488855 with exit 0; ModelSpec refuses the same file with E-CKPT-004. Truncation and a missing parameter name are refused. |

## Mutants (scratch copy of 97a78ac)

| mutant | unit suite | adversary cases |
|---|---|---|
| a: attention term dropped from the FLOP formula | killed (3) | killed |
| b: tokens/s and GFLOP/s over the mean | survives (8/8) | killed by `adv_tb_1` |
| c: key mask filled with zeros | killed (3) | killed |
| d: opponent GELU switched to erf | survives the loss check (5.8e-5 < 8.48e-4) | not testable |
| e1: per-sequence FLOP constant of one layer | killed (2) | killed |
| e2: tokens check uses 256·B | killed (2) | killed |

## Attacked and not broken

The FLOP formula (product-by-product recount equals 339,738,624·B; the per-site profile shows 21
H×H and 12 H×F matmuls per step; 3,060,705,792 = dense 2,315,255,808 + attention 686,292,992 +
59,156,992 elementwise). Head split, post-LN placement, 1/√d scaling, `mse` mean and SGD match the
`.mspec`. The opponent's `--batch` reader at B = 32 and B = 128. The bench loop against
`mlp_sweep/runs.rs` (warm-up, calibration, 3 runs, span, median, `dirty`, version). An unlisted
`--point 7`. `--steps 0` writes step 0 with the initial parameters and `infer` on it prints the same
loss. The key-mask rule (two masks both ones; a mask read elsewhere still E-DATA-001). The "Not
measured" section drops the block once it has a run. No `/home/` path, no Python outside
`opponents/`, no comments in touched config files.

```findings
- file: crates/modelspec-runtime/src/cpu/interpreter.rs
  line: 850
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: pre-existing
  message: the CPU mse's single f32 running sum is 1.56e-3 off its f64 value at 2,097,152 elements, so ModelSpec's step-0 loss at B = 128 (0.7482758) is outside the run tolerance of PyTorch's (0.74944544)
- file: opponents/transformer_block/README.md
  line: 69
  category: judgement
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the step-0 loss check at the run tolerance passes an erf GELU, a GELU coefficient of 0.04 and eps 1e-5, so it does not show that the two sides compute the same function
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 1701
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: a transformer-block run without tokens_per_step is accepted and drawn in the MLP sweep layout (hidden width H, row none) with no tokens-per-second column
- file: crates/modelspec-runtime/tests/transformer_block.rs
  line: 105
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the median test uses three equal runs, so taking the mean instead of the median leaves all 8 unit cases green
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 260
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the 128·B tokens_per_step check needs batch, so a transformer-block row with tokens_per_step 1000 and no batch is accepted
- file: crates/modelspec-cli/src/train.rs
  line: 257
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: --write-batch claims the layout of a dense --data directory, but --data on the example's written batch is refused with E-DATA-001
- file: opponents/transformer_block/transformer_block.py
  line: 115
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the opponent's .msckpt reader loads a checkpoint with a corrupted payload and prints loss 1.3488855 with exit 0 where ModelSpec refuses the same file with E-CKPT-004
```

## Resolution (coordinator, D84)

All seven are fixed in one pass before the timed runs. ADV-TB-5: the CPU oracle's full
reductions (`mse`, and every other loss sum over the whole tensor) accumulate in `f64` and round
once, so the CPU stays the accurate side at every batch; the two `adv_tb_5` cases become green
unit cases. ADV-TB-6: the opponent gains a one-step parameter comparison (`--compare-step`): from
the initial checkpoint and the written batch it takes one SGD step and reports the largest
absolute difference per parameter against ModelSpec's `--steps 1 --checkpoint`; the README names
that check, not the loss check, as what shows the two sides compute the same function, and the
mutant table above is its evidence. ADV-TB-2 and ADV-TB-3: a `transformer-block` row is refused at
the field when it lacks `batch` or `tokens_per_step`. ADV-TB-1: the median test uses distinct
runs. ADV-TB-4: the help text, CHANGELOG and DESIGN.md describe the file format, not a `--data`
directory. ADV-TB-7: the opponent's reader verifies the payload digest and refuses the corrupted
file as ModelSpec does. The adversary's three test files stay in the tree with the `#[ignore]`
attributes removed.
