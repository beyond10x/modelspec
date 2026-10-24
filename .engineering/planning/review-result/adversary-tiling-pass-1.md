---
format: aep.planning-md/3
id: review-result:adversary-tiling-pass-1
kind: review-result
status: active
title: Adversary pass 1 on matmul-shape-aware-tiling
relations:
- reviews: story:matmul-shape-aware-tiling
revision: 1
---
```
unit: story:matmul-shape-aware-tiling (head 8c4b085 plus my untracked test files)
verdict: green
cases: executed 1594→1601 (default suites), red 0; matmul device run (--ignored) 2→4
origin: n/a
wrote-outside-worktree: 1 directory, <wave-root>/tiling/adv1-scratch (listed below)
needs-coordinator: none
```

**What I touched:** `git --no-pager diff --stat` is empty because I changed no tracked file. `git status --short` shows only these untracked files, all tests or their fixture:
- `crates/modelspec-core/tests/adversary_tiling_rule.rs`
- `crates/modelspec-core/tests/adversary_tiling_stale_artifact.rs`
- `crates/modelspec-runtime/tests/adversary_tiling_device.rs`
- `crates/modelspec-runtime/tests/adversary_tiling_spirv.rs`
- `crates/modelspec-runtime/tests/fixtures/adversary_tiling_same_bits.txt` (digests I recorded at base 269b557)

**Attacks.** Each case was run alone first, before any suite run. Device runs used the local GPU with `MODELSPEC_VK_VALIDATION=1`.

| # | Hypothesis | Test or command | Output | Verdict |
|---|---|---|---|---|
| 1 | Bits change at the edges of the rule | `adversary_tiling_device.rs` `edge_matmuls_give_the_cpu_bits`: 34 shapes the fixture lacks (127 vs 128 blocks, `n` 63/64/65/127, `m` 1 and 2, `m % 64 = 1`, `k` 1/2/7/15/16/17/31/32/33/48/63/64/65/71/72/80/120/127, square sweep sites), three epilogues, normal inputs plus a family whose CPU sum is `-0.0` | `edge matmuls: 34 shapes, 194461912 elements compared bit for bit … ok` | green |
| 1 | Bits differ from the base commit's kernel | `edge_matmuls_write_the_bits_of_the_base_commit`: the 34 shapes, 5 large sweep sites and a subnormal family, against digests I recorded in a base copy | `base bits: 351 views compared … ok` | green |
| 1 | The unit's own fixture was recorded at HEAD, not base | Their `vulkan_matmul_same_bits` re-recorded in the base copy, then `diff` against the committed fixture | `FIXTURE-IDENTICAL` | green |
| 2 | Partial tiles read or write past a view | Same device test: inputs end in 64 NaN words, outputs in guard words, validation on | no element differs, 0 guard words changed, 0 validation messages | green |
| 3 | Sums not in ascending `k`, start from zero, or miss `NoContraction` | `adversary_tiling_spirv.rs` `every_register_blocked_chain_starts_at_the_k0_product_and_ascends_in_k`: follows each accumulator's `OpFAdd` chain in 132 modules (22 shapes × 3 epilogues × float controls on/off), plus `spirv-val`. Its control case `the_order_check_finds_a_chain_started_from_zero` proves the checker catches a mutated module | `132 modules read … ok`, control ok | green |
| 4, 5, 8 | Rule not total; tile over 16 KiB or 128 invocations; `cols_per_invocation` ≠ 1 outside the block rule; DESIGN text ≠ code | `adversary_tiling_rule.rs` (3 tests): 46³ boundary shapes plus 200,000 seeded shapes, checked against the rule rewritten from DESIGN.md § 5.2 and, for non-blocked shapes, against the base tile field for field | `3 passed` | green |
| 6 | An artifact compiled before the change still runs | Base binary compiled MNIST into scratch; `adversary_tiling_stale_artifact.rs` reads it with HEAD's reader | `E-ARTIFACT-003 … kernel ecca4089-… is not the id its content gives` | green (refused) |
| 7 | Every matmul site of a real training step | `train --backend vulkan` on sweep b1000 h1024 (20 steps) and b8192 h1024 (10 steps), HEAD binary vs base binary, sha256 of stdout (final parameters printed in full) | `319819d4… = 319819d4…`, `cb23eeb2… = cb23eeb2…` | green |
| 9 | Goldens or docs drift | `cargo run -q -p modelspec-docs -- generate --check`; `--test ir_kernel --test adversary_guide_kernel_lines` | `DOCS_EXIT=0`; 9 + 2 passed | green |
| 10 | The race fix is not atomic | Read only, no test: the rename stays in one directory of one filesystem; the thread id is unique in the process; the directory is private to the test binary; the content is the same for each `lr` | — | green by reading, not run |

**Suite runs, after my cases existed:**
- `b10x-build-slot cargo test --workspace`: EXIT=0, 1182 passed, 93 ignored. 5 of those are mine, so 1177 without them.
- `b10x-build-slot cargo test -p modelspec-runtime --features vulkan`: EXIT=0, 419 passed, 57 ignored. 2 are mine, so 417 without them.
- Device run of `vulkan_matmul_same_bits` and `adversary_tiling_device` with `--ignored` and validation on: EXIT=0, 4 passed.
- `cargo fmt --all` and clippy with `-D warnings` on modelspec-core and on modelspec-runtime with `--features vulkan`: clean.

**What I could not attack:**
- A hand-built `MatMulTile` cannot reach the plan verifier. Every user recomputes the tile from the shape (`schedule.rs:658`, `emit.rs:425`, `dispatch.rs:718`), so there was nothing to test.
- `k = 0` is refused by emit (`emit.rs:424`), so I did not test it.

**Two things to know about the tests I left:**
- `adversary_tiling_stale_artifact.rs` stays `#[ignore]` and fails without `ADVERSARY_TILING_BASE_ARTIFACT` pointing at an artifact compiled at base.
- `adversary_tiling_same_bits.txt` is tied to this device and driver (`10de:2204 driver_version 2559459584`). On any other one the test fails rather than compares, the same as the unit's own fixture.

**Written outside the worktree**, all under `<wave-root>/tiling/adv1-scratch`:
- `base/`: a copy of 269b557 including its `target/` build directory. Two of its test files were changed in the copy only. Per AGENTS.md, remove it with `cargo clean --target-dir …/base/target` when the unit merges.
- `art/mnist-base/`: the artifact compiled at base.
- `e2e/`: the two sweep sources and four outputs of about 21 MB each.
- `base-build.log`, `suite-workspace.log`, `suite-vulkan.log`, `suite-device.log`.

```findings
[]
```
