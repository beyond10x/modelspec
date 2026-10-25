---
format: aep.planning-md/3
id: review-result:adversary-sweep-idle-pass-1
kind: review-result
status: active
title: Adversary pass 1 on sweep-run-starts-busy
relations:
- reviews: story:sweep-run-starts-busy
revision: 1
---
Now I'll write the mutant-applying script (Rust is not required for a scratch shell helper outside the repo; it is a shell script in scratch).
unit: story:sweep-run-starts-busy, branch impl/sweep-run-starts-busy at 6fb6256 (worktree wt-558c2c3156cd)
verdict: findings 3 (blocker 0, warning 1, note 2)
cases: executed 21→26, red 0
origin: introduced 2 / pre-existing 0 / undecided 1
wrote-outside-worktree: 1 (the assigned scratch directory)
needs-coordinator: F3. The story's acceptance items idle-bisected and idle-reproduced cannot be met as written, because the diagnosis shows their premise is false. Accept that refutation or amend the acceptance.

**1. `git --no-pager diff --stat`**: empty. Nothing tracked was changed. The only new files are two untracked test files: `crates/modelspec-runtime/tests/adversary_sweep_idle.rs` and `adversary_sweep_idle_plan.rs`. `.engineering/planning/story/step-sites-beat-pytorch.md` is also untracked, but it is the coordinator's file and I did not touch it.

**2. Cases added** (written first, then run alone with `MODELSPEC_VK_VALIDATION=1`; all green on 6fb6256)

| case | asserts | now |
|---|---|---|
| `adv_series_at_the_widest_point_is_bit_for_bit_train_with` (device) | Series [20, 3, 261, 20] at b8192 h4096: each run equals `train_with` bit for bit, with every counter. It also checks host waits, submits and bytes up and down against values computed from the source's shapes. | green |
| `adv_series_at_the_widest_point_with_fast_math_is_bit_for_bit_train_with` (device) | The same with `fast_math` | green |
| `adv_series_counters_are_the_source_s_at_the_unit_s_point` (device) | The unit's own series, with those computed counters | green |
| `adv_a_series_run_that_diverges_is_the_refusal_train_with_makes` (device) | A series whose second run diverges ends with the same E-TRAIN-001 refusal, run and counters that `train_with` gives | green |
| `adv_a_point_s_plan_is_the_same_at_boundary_step_counts` (host) | All 12 points give the same plan at 1, 255, 256, 257, 1021 and 10,000,000 steps as at 20 | green |

**3. Suite run**, after the cases existed: `MODELSPEC_VK_VALIDATION=1 b10x-build-slot cargo test --release -p modelspec-runtime --features vulkan --test vulkan_mlp_sweep --test mlp_sweep --test adversary_sweep_idle --test adversary_sweep_idle_plan -- --include-ignored --test-threads=1` gave 26 passed, EXIT=0. The `before` figure of 21 is the same run with my two files left out (19 + 2).
- Every `--ignored` device test of the runtime crate, run with `--no-fail-fast` and validation on, is green.
- 6 unrelated cases fail there. All 6 are already-known `#[ignore]`d reds (F3, F4, N2, ADV-TOK-1) or need the real corpus.
- `cargo fmt --check` and clippy with `-D warnings`, both workspace-wide and runtime `--features vulkan --all-targets`, are clean.

**4. Findings**

| id | file:line | finding | how it was shown | what reaches it | verdict / origin / severity |
|---|---|---|---|---|---|
| F1 | `tests/vulkan_mlp_sweep.rs:146` | The series test compares `train_repeatedly` only with `train_with`, and both run through the same `execute_runs` body. A defect common to both is invisible to it. | Mutant e moves the per-run upload into its own submit and wait before the timed window. The unit's suite stays green. My case goes red: `left: (1, 1, 0, 13008980)` `right: (1, 1, 39026688, 13008980)` | Any edit to the shared run body. The repo's `vulkan_train.rs:205` literal would also catch this mutant, but it is not in the unit's suite. | CONFIRMED / introduced / note |
| F2 | `src/vulkan/runner.rs:935` | No test pins the end of the timed window to the host wait. | Mutant b moves `count(...)` after the host read-back. Both suites stay green: gaps `[12, 13]`, `UNIT_EXIT=0`, `ADV_EXIT=0`. I could not write a robust case for it from outside the runner. | The harness-honest acceptance item | CONFIRMED / undecided (the same mutant was not run against 74a9520) / note |
| F3 | `.engineering/planning/story/sweep-run-starts-busy.md` (Acceptance) | idle-bisected asks for "the commit that introduced the idle" and idle-reproduced asks for a dip-free block-kernel trace. `diagnosis.md` shows every commit dips, including 40dd797 interleaved (r1). The P0-versus-P3 idle state before each run is the stated reason (`r1-final` 1920 MHz P0, `r3-final` 780 MHz P3). | A read of the documents, not a program | The story's acceptance as written | CONFIRMED / introduced / warning |

**Mutants** (applied to a scratch copy; the tree under review was never edited)

| mutant | unit suite | my cases |
|---|---|---|
| a: parameter reset only on the first run | caught: `7 steps: the series run differs` | caught (all 4) |
| b: window ends after the host read-back | survives | survives |
| c: no host-cached preference in `download_buffer` | caught: `idled [49, 48] ms` | survives |
| c as briefed ("skip the invalidate") | cannot be built: `download_buffer` always requires `HOST_COHERENT`, so there is no invalidate to skip | n/a |
| d: 150 ms sleep between runs | caught: `[171, 167]` | survives |
| d24: 24 ms sleep | survives: `[37, 37]` | survives |
| e: upload moved before the window | survives | caught |
| old: 74a9520's harness, one `train_with` per run | caught: `[126, 122]` | survives |

**Re-measurement** (one session, GPU alone, clocks not locked, clean 6fb6256, records in `<scratch>/measure/`)

| side | µs per step, runs 1/2/3 | median | lowest SM clock after 1 s | gaps |
|---|---|---|---|---|
| modelspec | 6,890.0 / 6,912.6 / 6,943.5 | 6,912.6 | 1740 / 1710 / 1710 MHz | 27 / 28 ms |
| PyTorch eager | 6,317.8 / 6,334.8 / 6,333.4 | 6,333.4 | 1635 MHz in each run | 0 / 1 ms |

idle-removed holds: no sample is below 1,500 MHz and the median is under 7,188 µs. The implementor's own acceptance JSONL was recorded from a dirty tree (`commit 74a9520, dirty: true`). This clean run replaces it.

**Same bits**: base binary versus head, sha256 of stdout and checkpoint, for `mnist_mlp.mspec` and for b8192 h4096 with `--steps 20`. All four are equal (head 0d16ed3e…, 32f6858e…, 8e81e1cf…, 498c7686…) and match the unit's record.

**5. Tried and could not break**
- Each run starts from step 0, checked at every step count and at the widest point.
- A series that diverges ends correctly.
- The plan does not change with the step count.
- Validation-layer runs are clean.
- Same bits hold.
- The 40 ms bound would catch the brief's 60 ms regression: the d24 plant adds about 24 ms and gives 37 ms in the test, so 60 ms in the bench would show as about 47 ms.
- Calibration matches 74a9520's logic.
- The bench's JSONL still reports figures per run.

**6. Written outside the worktree**: `<wave-root>/sweep-idle/adv1-scratch` (logs, mutant scripts, the base and mutant source trees, same-bits outputs, measurement records). Both scratch `target/` directories were removed with `cargo clean`; 230 MB remains.

`git status --short`:
```
?? .engineering/planning/story/step-sites-beat-pytorch.md
?? crates/modelspec-runtime/tests/adversary_sweep_idle.rs
?? crates/modelspec-runtime/tests/adversary_sweep_idle_plan.rs
```

```findings
- file: crates/modelspec-runtime/tests/vulkan_mlp_sweep.rs
  line: 146
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the series test compares train_repeatedly only with train_with through the same execute_runs body, so moving the per-run upload outside the timed window (mutant e) passes the unit's suite
- file: crates/modelspec-runtime/src/vulkan/runner.rs
  line: 935
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: undecided
  message: no test pins the timed window's end to the host wait; moving the count after the host read-back survives every suite
- file: .engineering/planning/story/sweep-run-starts-busy.md
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: acceptance items idle-bisected and idle-reproduced ask for an introducing commit and a dip-free block-kernel trace, which the diagnosis shows do not exist, so the coordinator must accept the refutation or amend the acceptance
```
