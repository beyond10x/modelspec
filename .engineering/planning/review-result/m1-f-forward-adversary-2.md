---
format: aep.planning-md/3
id: review-result:m1-f-forward-adversary-2
kind: review-result
status: active
title: 'm1-f unit 1 adversary pass 2: red on the size-bound slack'
relations:
- reviews: story:m1-f-forward-matmul-tiled
revision: 1
---
```
unit: story:m1-f-forward-matmul-tiled, ab375a5 plus 3 untracked adversary test files in modelspec-impl-m1f-fwd2
verdict: red
cases: executed 1228→1231, red 1
origin: introduced 1, pre-existing 0, undecided 0
wrote-outside-worktree: 6 files under ~/.cache/modelspec-wave-m1-f/m1-f-forward-matmul-tiled/scratch/adversary2/
needs-coordinator: no
```

The pass is red on one case, and that case is about the unit's size-bound test, not the kernel. Every bit-for-bit device case I added is green: the two-row threshold, the partial last row block and every remainder of the tile loop. Disk stayed above the floor throughout (67 to 78G free).

1. **Diff.** `git --no-pager diff --stat` is empty because the 3 files are new and untracked. All three are test files and nothing else changed:
   - `crates/modelspec-core/tests/adversary2_matmul_tile.rs`
   - `crates/modelspec-runtime/tests/adversary2_matmul_tiled_spirv.rs`
   - `crates/modelspec-runtime/tests/adversary2_vulkan_matmul_tiled.rs`

2. **Cases added, each run alone before the suite:**

| Case | What it asserts | Result |
|---|---|---|
| `adversary2_matmul_tiled_spirv.rs` `a_large_k_matmul_module_stays_within_four_constants_of_the_sixteen_tile_bound_at_more_shapes` | The same measure as `spirv_val.rs:732`, over 7 shapes and every remainder of `k` modulo 4 tiles at 3 large values of `k` | **red**, 30 s |
| `adversary2_vulkan_matmul_tiled.rs` `the_two_row_threshold_gives_the_cpu_bits_on_both_sides` | Rows switch at 255/256/257 one-row workgroups (n = 16, 3, 1, 17); bits equal the CPU on signed normal inputs, 3 epilogues | green, 10 s |
| `…` `a_two_row_last_block_with_some_second_rows_inside_gives_the_cpu_bits` | Last row block where `m % block > block/2` (2045, 2057, 1021, 8185, 33000); bits equal the CPU | green, 4 s |
| `…` `the_tile_loop_gives_the_cpu_bits_at_every_remainder_of_its_passes` | 11 to 18 full tiles × partial last tile of 0, 1 and depth − 1, for a one-row tile (m 5) and a two-row tile (m 2045); 48 shapes; bits equal the CPU | green, 50 s |
| `adversary2_matmul_tile.rs` `a_matmul_tile_takes_two_rows_from_exactly_256_one_row_workgroups` | Row count and workgroup count written as literals at 254/255/256/257 | green |

   The device cases ran with `MODELSPEC_VK_VALIDATION=1`, one at a time, on the RTX 3090. `nvidia-smi` showed no other compute process (307 MiB in use). The threshold test closes a mutant the suite misses: change `>=` to `>` in `MatMulTile::of` and the existing tests still pass, because no shape in `kernel_plan.rs` has exactly 256 one-row workgroups. I worked that out from the shapes and did not build the mutant.

   The red output of the size case, verbatim:
   ```
   [208, k] x [k, 2] depth 16: up to 16 tiles (16935, 3056), worst large k 10031 gives 16955 words, id bound 3061 (+5 ids, +20 words)
   [176, k] x [k, 2] depth 16: up to 16 tiles (16935, 3056), worst large k 10031 gives 16955 words, id bound 3061 (+5 ids, +20 words)
   ```
   At the other shapes the excess is at most 4: `(704,16)`, `(704,8)` and `(352,416)` sit exactly at +4, `(100,64)` is +3 and `(5,3)` is +2.

3. **Suite run, after the cases existed:** `cargo test --workspace --no-fail-fast`, then `cargo test -p modelspec-runtime --features vulkan --no-fail-fast`. Both exited 101. 1,229 passed, 2 failed, 90 ignored. The 2 failures are my size case, which runs in both commands. The unit's own `a_matmul_module_is_no_larger_at_any_k_than_at_a_k_of_sixteen_tiles` passed in both.
   - I did not run the suite a second time without my files. I got `<before>` by subtracting my 3 executed cases from 1,231: the spirv case twice and the core case once. The 3 device cases are ignored and are not in either count.

4. **Finding:** `crates/modelspec-runtime/tests/spirv_val.rs:729`, the slack `SPARE_CONSTANTS = 4`. It is CONFIRMED and introduced by the unit.
   - **What was measured:** at n = 2 (depth 16), a large `k` adds 5 ids and 20 words over the largest module at `k` up to 16 tiles. At small `k`, the offsets of the tiles read after the loop coincide with constants the module already holds; at large `k` they do not, and the loop's trip count is one more constant. At m = 176 or 208, `m` itself is one of those offsets. At n = 2, 192 and 224 are already b-offsets of the tiles before the loop and of the loop body, and 384 is a staging offset.
   - **What reaches it:** any `[m, k] x [k, 2]` matmul with a large `k`, for example the weight gradient of a two-output layer over a large batch. The module is still bounded: it is about 17k words, nowhere near the SPIR-V id limit of 4,194,303, and the code does not grow with `k`. What is wrong is the test's stated bound, and it is green only because its shapes miss this case.
   - **Fix to name:** derive the slack from the number of read offsets that can coincide (2 per straight read, up to 4 reads, plus the trip count), or measure the reference at a large `k` with the same remainder.

5. **Attacked and not broken:**
   - Carried reads and the 4-tile pass logic, at every remainder 0 to 3, with and without a partial last tile, for one-row and two-row tiles: bit for bit.
   - The two-row threshold at 255/256/257 one-row workgroups for 16, 4 and 1 columns, and the n = 17 case, with bits equal to the CPU.
   - A last row block with mixed valid and invalid second rows, an odd `m`, and the row past the end in the last workgroup: bit for bit.
   - The SPIR-V id bound: bounded at any `k` up to `u32::MAX / max(m, n)`.
   - Pass 1's `m` change in `a_matmul_over_more_than_65535_workgroups…`: `m = 16·65536 + 9` still gives 65,537 workgroups on a 2-D grid, so it still tests what pass 1 meant.

6. **Paths written outside the worktree**, all under `~/.cache/modelspec-wave-m1-f/m1-f-forward-matmul-tiled/scratch/adversary2/`:
   - `spirv-case.log`, `suite.log`, `helpers.rs.txt`
   - `dev-the_two_row_threshold_gives_the_cpu_bits_on_both_sides.log`
   - `dev-a_two_row_last_block_with_some_second_rows_inside_gives_the_cpu_bits.log`
   - `dev-the_tile_loop_gives_the_cpu_bits_at_every_remainder_of_its_passes.log`

   No processes are left running, and I released my worktree lease.

7.
```findings
- file: crates/modelspec-runtime/tests/spirv_val.rs
  line: 729
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: at [m, k] x [k, 2] (m 176 or 208, depth 16) a large k emits 5 more ids and 20 more words than the largest module at k up to 16 tiles, so the size-bound test's slack of 4 constants is wrong and stays green only because its shapes miss n = 2; the module itself stays bounded
```
