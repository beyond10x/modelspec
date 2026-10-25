---
format: aep.planning-md/3
id: review-result:adversary-step-sites-pass-1
kind: review-result
status: active
title: Adversary pass 1 on step-sites-beat-pytorch
relations:
- reviews: story:step-sites-beat-pytorch
revision: 1
---
All nine mutants are done. Now the cleanup and the final tree state.
unit: story:step-sites-beat-pytorch (wave perf-1, modelspec, head 91e9bbd over 3ef38aa)
verdict: findings 4 (blocker 0, warning 0, note 4)
cases: executed 1659→1663, red 1
origin: introduced 4, pre-existing 0
write-outside-worktree: yes, everything under <wave-root>/step-sites/adv1-scratch/
needs-coordinator: no

All four findings are notes and none is a blocker. Every bit check matched, all 9 mutants were killed, and my one timed round came within 2.8% of the recorded medians.

**`git --no-pager diff --stat`**: empty. `git status --short`:
```
?? crates/modelspec-core/tests/adversary_step_sites.rs
?? crates/modelspec-runtime/tests/adversary_step_sites_device.rs
```

**Cases added**
- `<trees>/wt-2c5a297a8e42/crates/modelspec-core/tests/adversary_step_sites.rs`: 4 green tests over 9 probe models (epilogue rule, epilogue before a parameter's update, where the loss sum goes, loss steps), plus the red ignored test `adv_s1_f1_relu_grad_of_an_epilogue_activation_is_an_epilogue_as_design_states`.
- `<trees>/wt-2c5a297a8e42/crates/modelspec-runtime/tests/adversary_step_sites_device.rs`: 1 device test, `#[ignore = "needs a Vulkan device"]`. It checks that the probe models train to the CPU bits, the dispatch counts, that infer loss equals final loss, and that validation is clean. It is green on the device (0.81 s).

Red output, run alone:
```
thread 'adv_s1_f1_relu_grad_of_an_epilogue_activation_is_an_epilogue_as_design_states' (84592) panicked at crates/modelspec-core/tests/adversary_step_sites.rs:270:9:
assertion `left == right` failed: relu_of_relu
  left: None
 right: Some(ReluGrad)
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out; finished in 0.01s
```

**Suite with my files present**

| command | passed | failed | ignored | exit |
|---|---|---|---|---|
| `cargo test --workspace --no-fail-fast` | 1663 | 0 | 143 | 0 |
| `cargo test -p modelspec-runtime --features vulkan --no-fail-fast` | 649 | 0 | 95 | 0 |
| `task docs-check` (`modelspec-docs generate --check`) | current | | | 0 |

**Hypotheses**

| # | probe | observation | verdict |
|---|---|---|---|
| 1 epilogue rule | 9 probe models, plan checks plus device bits | Correct pre-activation read on all of them, including `relu(w1)` and two epilogues sharing one pre-activation. The exception is `relu(relu(matmul))`: no epilogue | F1 |
| 2 bits | MNIST train, checkpoint and infer digests, head against base 3ef38aa, GPU empty both times | identical (checkpoint 32f6858e…e608, infer 33bdce54…3f89) | holds |
| 3 loss placement | wide second layer, ties, transposed, scaled loss, plus `--profile` | the sum goes after the first of the equally largest matmuls, behind a barrier; `verify_plan` passes | holds; ties covered only by my test (F3) |
| 4 infer and final loss | CPU against Vulkan, against base | equal | holds |
| 5 checkpoint resume | stop at 40, run 60 more, compare with 100 straight (MNIST, deep3, two relus) | bit-equal, also to the CPU run | holds |
| 6 descriptors and spirv-val | counted by hand; spirv_val | 29 descriptors in 9 sets, 6 barriers, 12 buffers; 144 modules valid, 36 of them ReluGrad | holds |
| 7 judgement numbers | one timed round each side, GPU empty | modelspec 7,033.3 µs (+2.5%), eager 6,750.0 µs (+2.8%), ratio 1.042 against 1.044 recorded | holds; recorded command fails (F2) |
| 8 verdict and guide | medians recomputed from the jsonl; guide clocks and kernel lines | all match; "goal missed" is stated correctly | holds; opponent row lacks its dirty note (F4) |
| 9 mutants | table below | 9 of 9 killed | holds |
| 10 docs | spec-validate, both schema-checks, docs-check | all exit 0 | holds |

**Findings**
- **F1:** `<trees>/wt-2c5a297a8e42/crates/modelspec-core/src/schedule.rs:378` has `&& !absorbed.contains(&x)`. That line refuses the epilogue when `x` is itself a relu-epilogue activation. DESIGN.md:180 ("when `x` is stored before the matmul") and the CHANGELOG say the epilogue should apply there. Bits and dispatch counts are unaffected: the `ReluGrad>ReluGrad@1` chain gives the same counts. Red output is quoted above.
- **F2:** `benchmarks/raw/2026-10-25-step-sites/verdict.md:14` records `cargo bench … --bench mlp_sweep -- --bench --point 8192:4096 --min-seconds 6`. Run as written it fails:
  ```
  error: the argument '--bench' cannot be used multiple times
  ```
  It is the only file with this form. Without the second `--bench` it runs, and that is how I measured.
- **F3:** mutant g (`flop >= most`, which picks the last of equal matmuls) passes the unit's own suite. Only my green test `the_loss_sum_sits_after_the_first_largest_matmul_and_behind_its_inputs_barrier` catches it. The fix is to keep that test.
- **F4:** `benchmarks/runs.yaml:2216` (the eager row) gives commit 3ef38aa, but its jsonl says `"dirty":true` and the row has no note. Earlier rows in that file add a note when the opponent record is dirty. The modelspec row says so in its `source` field.

**Mutants**, run against the scratch copy, logs in `adv1-scratch/mut/logs/`:

| mutant | killed by |
|---|---|
| a `x >= 0` | only device tests: vulkan_fma_oracle, vulkan_matmul_plan_sites_same_bits |
| b reads the activation | kernel_plan, adversary_fuse1_plan_semantics, step_sites_plan, ir_kernel, adversary_step_sites |
| c another reader exists | fuse_chains, adversary_fuse |
| d1 sum one place earlier | step_sites_plan, ir_kernel, adversary_step_sites |
| d2 sum never moved | the same three |
| e loss steps include gradients | step_sites_plan, adversary_step_sites |
| f pool of 31 | runner lib: `the_pool_counts_every_read_and_write_of_the_mnist_plan`, `a_two_write_step_takes_a_descriptor_per_write` |
| g last of equals | adversary_step_sites only (F3) |
| i loss steps drop a step | step_sites_plan, vulkan_train, train_vulkan, both adversary files |

**GPU checks**
- `adv1-scratch/timed/apps-before.txt`: empty at 11:17:47Z, 11:18:02Z and 11:20:15Z.
- Busy SM clocks: 1650–1950 MHz for eager (`clocks.csv`) and 1680–1935 MHz for modelspec (`clocks2.csv`).
- Clocks never locked, no fast_math.

**Choices I made**
- The build-slot budget was set to 52, then 60, then 72 instead of 48. The repository's footprint grew from 49G to 65G while other lanes built.
- I made a scratch-only `git init` in `adv1-scratch/mut` so mutants could be reset.
- Both scratch targets are cleaned and the lease is ended.

**Written outside the worktree**: everything is under `<wave-root>/step-sites/adv1-scratch/`, 68M in total: `probe/`, `base/`, `mut/` (with `logs/` and its scratch repo), `timed/`, `logs-suite-*.log`, `logs-docs-check.log`, `run.sh`, `mutate.sh`, `mutsuite.sh`, `mutloop.sh`, `timed.sh` and `timed2.sh`.

```findings
- file: crates/modelspec-core/src/schedule.rs
  line: 378
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: DESIGN.md:180 and the CHANGELOG make a matmul-gated relu_grad with a stored x an epilogue, but schedule.rs refuses it when x is a relu-epilogue activation (relu(relu(matmul))); red case adv_s1_f1_relu_grad_of_an_epilogue_activation_is_an_epilogue_as_design_states
- file: benchmarks/raw/2026-10-25-step-sites/verdict.md
  line: 14
  category: reproducibility
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the recorded bench command passes --bench after --, and cargo bench adds its own, so it exits 2 with "the argument '--bench' cannot be used multiple times"
- file: crates/modelspec-runtime/tests/step_sites_plan.rs
  line: 223
  category: test-gap
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: mutant g (flop >= most, last of equal matmuls) survives the unit's suite; only adversary_step_sites catches it; position 3 of this test's id list compares the plan with itself
- file: benchmarks/runs.yaml
  line: 2216
  category: record-accuracy
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the eager row gives commit 3ef38aa although its raw record says dirty true, and it has no note, unlike earlier rows with dirty opponent records
```
