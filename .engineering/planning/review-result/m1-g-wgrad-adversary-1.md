---
format: aep.planning-md/3
id: review-result:m1-g-wgrad-adversary-1
kind: review-result
status: active
title: 'm1-g unit 1 adversary pass 1: 0 red, 2 notes'
relations:
- reviews: story:m1-f-weight-grad-matmul-tiled
revision: 1
---
```
unit: story:m1-f-weight-grad-matmul-tiled, commit e47e807 (base 2449897) plus 2 untracked test files
verdict: nothing found that blocks: no red case; 2 judgement notes
cases: executed core 314→317, runtime device 8→10 (on top of 323 runtime non-ignored); red 0
origin: introduced 0 / pre-existing 1 / undecided 1
wrote-outside-worktree: 7 paths under ~/.cache/modelspec-wave-m1-g/m1-f-weight-grad-matmul-tiled/scratch/adversary/
needs-coordinator: none
```

1. **Diff.** `git --no-pager diff --stat` is empty; both new files are untracked test files:
   - `crates/modelspec-core/tests/adversary_kernel_two_row.rs`
   - `crates/modelspec-runtime/tests/adversary3_vulkan_matmul_tiled.rs`

2. **Cases added, each run alone before any suite:**

| File | Test | What it asserts | Now |
|---|---|---|---|
| `adversary_kernel_two_row.rs` | `every_one_row_tile_is_the_base_commits_tile` | every one-row tile on a grid of 4,840 shapes equals the rule at `2449897`, copied verbatim | green |
| same | `every_two_row_tile_reads_2_ahead_at_the_deepest_depth_up_to_16_that_fits` | every two-row tile: ahead 2; depth ≤ min(k,16), lowered only when doubling it overflows 16 KiB; ≤ 16 KiB; buffers = 2 exactly when there is more than one tile | green |
| same | `two_row_tiles_whose_depth_the_memory_limit_lowers` | literal (depth, buffers) for 1 and 2 columns, e.g. `(32641,100,1)→(4,2)`, `(16321,17,2)→(8,2)`, `(32641,15,1)→(15,1)` | green |
| `adversary3_vulkan_matmul_tiled.rs` | `two_row_tiles_of_16_8_and_4_give_the_cpu_bits_at_0_to_10_full_tiles` | device result bit-equal to the CPU for all 3 epilogues: depth 16, 8 and 4, 0 to 10 full tiles, remainder 0, 1 and depth−1 | green: 96 shapes, 12,544,384 elements, 0 failing |
| same | `the_weight_gradient_rows_give_the_cpu_bits_at_k_around_100` | `[784,k]x[k,64]`, 18 values of k from 2 to 128, bit-equal | green: 3,612,672 elements, 0 failing |

   A first core run was red on an assertion of the reviewer's own (a two-row depth never exceeds the
   base's; `[8161,63]x[63,3]`: 16 against 15). Nothing claims that; the assertion was dropped.

3. **Suite runs, after the cases existed:**
   - `cargo test -p modelspec-core`: exit 0, 20 result lines, passed 317, failed 0.
   - `cargo test -p modelspec-runtime --features vulkan`: exit 0, passed 323, failed 0, ignored 45.
   - `MODELSPEC_VK_VALIDATION=1 … -- --ignored --test-threads 1`, all exit 0: `vulkan_matmul_tiled`
     2 passed, `adversary_vulkan_matmul_tiled` 3 passed, `adversary2_vulkan_matmul_tiled` 3 passed,
     `adversary3_vulkan_matmul_tiled` 2 passed.
   - Acceptance 2: `target/release/modelspec train examples/mnist_mlp.mspec --backend vulkan | sha256sum`,
     2 runs, both `b9bb028a6d25cec5829f76919e5e478abc4d71098227a859403bdf84b6826904`.

4. **Judgement findings, on `e47e807`:**

| file:line | Severity | Verdict | Origin | Finding |
|---|---|---|---|---|
| `crates/modelspec-core/src/ir/kernel.rs:359` | note | CONFIRMED | pre-existing | `Kernel::signature` hashes workgroup size and workgroups but not depth or ahead, so `k11` keeps id `d7d3124e` while its SPIR-V changed; profile lines before and after name one id for two modules |
| `CHANGELOG.md:14` | note | CONFIRMED | undecided | CHANGELOG gives `k11` 4.91 → 4.58 µs (medians of 3); DESIGN.md:273 and `kernel.rs:160-163` give 4.93 → 4.67 µs (means of 2) for the same change |

5. **Attacked, not broken:** one-row tiles unchanged (exhaustive grid against the base rule); the
   16 KiB limit for every two-row grid shape; the emitter at depth 16, 8 and 4 and at k = 1, 2, 3,
   15, 17, 31 to 33 and 47 to 49; the tile loop at every remainder; the partial last row block and
   the two-row threshold; the validation layer on every device run; the 1,800-step stdout hash;
   DESIGN.md § 5.2 wording against the code.

```findings
- file: crates/modelspec-core/src/ir/kernel.rs
  line: 359
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: the kernel signature omits tile depth and ahead, so k11 keeps module id d7d3124e across a change of its SPIR-V and profile lines before and after cannot be told apart by id
- file: CHANGELOG.md
  line: 14
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: undecided
  message: CHANGELOG reports k11 4.91 to 4.58 us (medians of 3) while DESIGN.md:273 and kernel.rs:160 report 4.93 to 4.67 us (means of 2) for the same change
```
