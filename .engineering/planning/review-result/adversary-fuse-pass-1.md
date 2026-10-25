---
format: aep.planning-md/3
id: review-result:adversary-fuse-pass-1
kind: review-result
status: active
title: Adversary pass 1 on fuse-elementwise-chains-and-gradient-sums
relations:
- reviews: story:fuse-elementwise-chains-and-gradient-sums
revision: 1
---
```
unit: story:fuse-elementwise-chains-and-gradient-sums
verdict: green
cases: executed 1461→1463, red 0
origin: n/a
wrote-outside-worktree: <wave-root>/fuse/adv1-scratch/{base,mut,orig,fixtures,runs,base-build.log,suite-before.log,suite-after.log,device-before.log,device-after.log}
needs-coordinator: b10x-build-slot refused with "trees/b10x/modelspec holds 42G, budget 40G" (the worktree target holds 15G; the other trees are not mine). I ran every build with the override the refusal names, B10X_BUILD_REPO_BUDGET_G=48, with 134G to 150G free on /
```

The `executed` counts are workspace tests, default features. Before is `--skip` of my 2 core cases: 1461 passed. After: 1463 passed, 0 failed, 122 ignored. The device suite went 69→71 passed, with 5 failures identical in both runs. Those 5 are `#[ignore]`d tests this unit does not touch: two need real dataset releases (env vars unset), two are gelu findings F3/F4, one is ADV-TOK-1, kept by design.

`git --no-pager diff --stat` prints nothing, because nothing tracked changed. `git status --short`:
```
?? benchmarks/raw/2026-10-25-fuse/adversary/
?? crates/modelspec-core/tests/adversary_fuse1_plan_semantics.rs
?? crates/modelspec-runtime/tests/adversary_fuse1_chain_device.rs
```
The raw directory holds `runs.txt`, `mutants.txt`, `suites.txt` and `fixtures/*.mspec` (10 models).

| # | Hypothesis | Test / evidence | Command | Quoted output | Verdict |
|---|---|---|---|---|---|
| 1 | A chain swallows a value someone else reads (read twice, a reduction, a matmul after an epilogue, loss or prediction, a parameter leaf, a transposed operand) | `adversary_fuse1_plan_semantics.rs`: `every_fused_plan_computes_the_terms_of_its_training_ir` (runs the plan symbolically against IR terms), `a_value_read_twice_keeps_its_dispatch`. Plus base vs fused binary on 10 fixtures, 200 steps each | `cargo test -p modelspec-core --test adversary_fuse1_plan_semantics`; `<bin> train --backend vulkan <f> --checkpoint …` | `2 passed`; 10 of 10 fixtures `SAME` (stdout and msckpt sha) | green |
| 2 | An adjoint sum is not added left to right, or `MAX_CHAIN_READS` overruns the bindings | Same core test (h read by 2 to 8). `vulkan_the_plans_adjoint_sum_of_four_adds_left_to_right` feeds 1e8, 1, -1e8, 1: left to right gives 1, the reverse gives 0 | `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_fuse1_chain_device -- --ignored --nocapture` | `adjoint: 90 elements, 0 not 1, validation on with 0 messages`; adj4/5/7 base vs fused `SAME` | green |
| 3 | Deterministic chains are not host bits, or fast_math leaves tolerance | `vulkan_artifact_chains_write_host_bits_and_fast_math_stays_within_tolerance` | same | `Sub>Mul@0>Add@0 n=16448: deterministic vs host differ 0, fast_math vs deterministic differ 2624, outside tolerance 0` | green |
| 4 | Goldens, `--record` stream or ModelGraph change beyond the fusion | Golden diff read; record runs (mnist, 20 steps) | `train --record` with both binaries | `23` lines each, byte-equal with the `*_id` fields removed | green |
| 5 | Resuming across the two binaries changes the result | mnist runs 100+100 vs 200, all four crossings | `train --resume … --steps 100` | every 200-step checkpoint `8e241b7c…`; both 100-step ones `c77b87f6…` | green |
| 6 | The counting rule or the profile breaks on chains | param_leaf `--stats --profile` | `train … --steps 40 --stats --profile` | `flop 270 bytes 1440` (2880 unfused minus 4×360 of intermediates); `dispatches per step: 9`, `uncounted_sites 0` | green |
| 7 | The guide's figures are wrong | Arithmetic checked; `adversary_guide_kernel_lines`; docs-check | as named | `2 passed`; docs-check `EXIT=0` | green |
| 8 | Mutants survive the unit's suite | 6 mutants on the scratch copy (`mutants.txt`) | `cargo test` on the copy | M1, M2, M4, M5, M3, M3b all killed by the unit's suite. M3b reproduces DESIGN's `32,987 of 661,520` | green |

`infer` cuts the step plan at the last step that writes the loss buffer (`runner.rs:1030-1035`). A chain can only end at the loss, never pass through it, so the cut is unchanged. On 6 fixtures, base vs fused `infer --backend vulkan` gave the same output each time. In pred_chain, CPU and Vulkan differ by 1 ulp in the loss (`0.23392996` vs `0.23392995`), on the base binary as well. That is within the tolerance DESIGN § 8 allows `infer`, so it is not a finding.

`cargo fmt --all` reformatted only my two files. clippy `-D warnings` exits 0 on modelspec-core, which has no vulkan feature, and on modelspec-runtime with `--features vulkan`. I cleaned both scratch targets (352.5 MiB and 4.9 GiB removed). I sent the lease `session-end`. These tests show no defect; they do not show that the change is correct.

```findings
[]
```
