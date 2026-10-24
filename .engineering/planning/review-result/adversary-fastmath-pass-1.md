---
format: aep.planning-md/3
id: review-result:adversary-fastmath-pass-1
kind: review-result
status: active
title: Adversary pass 1 on fast-math-mode
relations:
- reviews: story:fast-math-mode
revision: 1
---
```
unit: story:fast-math-mode, worktree impl/fast-math-mode at ad3f734 plus my 3 untracked test files
verdict: CONFIRMED (no blocker): 2 red cases, both pre-existing; nothing introduced by the diff went red
cases: executed 1614→1619, red 2
origin: introduced 0 / pre-existing 2 / undecided 1
wrote-outside-worktree: 1 directory (part 6)
needs-coordinator: ADV-FM-2, decide whether "run record" in ADR-008 and the acceptance means the --record file or the --stats line
```

**1. Diff**

`git --no-pager diff --stat` lists only the coordinator's uncommitted files, which I left alone: `.engineering/planning/decision-blocker/cooperative-matrix-f32-operands.md`, `.engineering/planning/story/fast-math-mode.md` and `CHANGELOG.md` (3 files, +9 −4). I added three untracked test files and touched nothing else:
- `crates/modelspec-runtime/tests/adversary_fastmath_emission.rs`
- `crates/modelspec-cli/tests/adversary_fastmath_cli.rs`
- `crates/modelspec-cli/tests/adversary_fastmath_device.rs`

**2. Cases (each run alone first)**

| Attack | Case | Result |
|---|---|---|
| 1, 2, 4 | `every_kernel_kind_of_a_fast_math_artifact_is_its_deterministic_module_less_no_contraction`: 7 sources × 2 profiles; no NoContraction; execution modes, capabilities and opcode counts unchanged; keys differ only in `fast_math` | green. Kinds checked: Broadcast 4, Elementwise 32, LossSum 10, MatMul 44 (register-blocked 18, epilogue 8), ReduceStage1/2 4/4, SgdUpdate 22, Transpose 26. 106,818 decorations dropped |
| 4 | `fast_math_and_deterministic_artifacts_differ_only_in_fast_math_and_no_contraction` (MNIST and sweep b8192 h4096 through the CLI) | green: "14 kernels, 8 module files differ" for each |
| 3 | `a_flipped_fast_math_is_refused_with_or_without_a_new_seal`: flip in the manifest, the first key, the last key, and deterministic→true | green: E-ARTIFACT-002 unsealed, E-ARTIFACT-003 resealed |
| 3 | `a_fast_math_artifact_relabelled_deterministic_throughout_is_refused` | **red, now `#[ignore]` ADV-FM-1**: `the reader accepted the tampered artifact …/relabelled` |
| 5 | `fast_math_without_a_backend_or_on_infer_is_a_usage_error_that_writes_nothing` | green |
| 6, 9 | `fast_math_runs_repeat_their_bits_across_an_intervening_deterministic_run`: fast, fast, deterministic, fast | green: same bytes each fast run; the deterministic run prints different bytes; all print `numerics class: tolerance` |
| 9 | `the_run_record_of_a_fast_math_run_names_the_class_tolerance` | **red, now `#[ignore]` ADV-FM-2**: `the run record names no numerics class:` and the `--record` lines carry only `RunStarted{"steps":2,"start_step":0}`, `StepCompleted`, `RunFinished` |

**3. Suite (run after the cases existed)**
- `b10x-build-slot cargo test --workspace` then `… -p modelspec-runtime --features vulkan`: EXIT=0 and EXIT=0, 1619 passed, 0 failed, 155 ignored.
- The "before" figure comes from that same run with my files' executions taken out: 3 in the cli file, plus the emission case, which ran in both invocations. 1619 − 5 = 1614.
- Device run: `MODELSPEC_VK_VALIDATION=1 … --release -p modelspec-cli --test train_fast_math --test adversary_fastmath_device --test train_vulkan -- --ignored --skip the_run_record`, EXIT=0, 5 passed. `nvidia-smi --query-compute-apps` listed 0 apps before every device run.
- `cargo fmt --all` touched only my files. Clippy with `-D warnings` is clean on `modelspec-cli` and on `modelspec-runtime --features vulkan`.

**4. Findings that are judgement, not a failing case (tree ad3f734)**
- **ADV-FM-1**, `crates/modelspec-core/src/artifact.rs:437`: the reader checks only the SPIR-V header, so a module whose bytes don't match its manifest's settings passes once resealed.
  - Pre-existing: a probe at base 2ab8c0b stripped the NoContraction decorations from 8 modules, rewrote their hashes, resealed, and got `read ok = true`.
  - Nothing reaches it: no command runs an artifact directory, and `seal` says outright that it does not authenticate. INFEASIBLE.
- **ADV-FM-2**, `crates/modelspec-observatory/src/event.rs:128`: ADR-008 (DESIGN.md:623) says "The run record and the `numerics class` line of `--stats` name the class". The unit meets only the `--stats` half.
  - Pre-existing: the base text says the same thing, and the diff does not touch the observatory.
  - It is reached by the documented `train --record`.
- **ADV-FM-3**, DESIGN.md:622 against `emit.rs:613`: ADR-008 forbids reassociation, but its only mechanism is removing NoContraction. As I remember the SPIR-V spec (not checked here, no local copy), that decoration also forbids reassociation, so removing it allows the driver to reassociate.
  - Mutant B (loss sum added in reverse order under fast_math only) left `train_fast_math` green, with the largest loss-tolerance share at 0.004305. Only `fast_math_emission.rs` caught it, and only because the emitter itself changed.
  - Reassociation done by the driver is caught by neither test. Undecided: it is an ADR question, not this diff.

**5. Attacked and not broken**
- Every kernel kind loses all its NoContraction decorations and nothing else changes; the float-control modes are identical with and without the flag.
- With the flag off, the base and head binaries write byte-identical artifacts (`diff -r`) for MNIST, linear regression and the sweep source.
- No kernel or pipeline cache exists (`PipelineCache::null()`, `pipeline.rs:208`), so a cache collision is impossible.
- `train` has no artifact input, so "an artifact compiled with the flag, run without it" cannot happen.
- The tolerance oracle: mutant A (loss divisor n+1) went red at share 9.69, and mutant C (SGD step ×1.01) went red at share 31.59.
- The step 11 module (transpose `289ddcbb`) and step 14 (`9c26b26f`) are byte-identical with and without the flag, so their slowdown is GPU clock state. The three slower small matmuls really do have different modules.
- Docs check: 0 drift. The goldens, `ir_kernel.rs` and `adversary_guide_kernel_lines.rs` are unchanged since base.

**6. Written outside the worktree**
- `<wave-root>/fastmath/adv1-scratch/` holds a base copy (`target/` 999M, a probe test and a blake3 dev-dep added only in that copy), a mutant copy (`target/` 1.5G; `emit.rs` currently carries mutant C), `emit.rs.head`, `mutant-{a,b,c}.diff`, `default-unchanged/` and `*.log`.
- Per AGENTS.md, both `target/` directories come out with `cargo clean --target-dir <dir>` when the unit merges.

```findings
- file: crates/modelspec-core/src/artifact.rs
  line: 437
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: pre-existing
  message: the reader accepts a fast_math artifact relabelled deterministic in its manifest and every key and resealed, because it checks only module headers and nothing executes an artifact directory
- file: crates/modelspec-observatory/src/event.rs
  line: 128
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: pre-existing
  message: the --record run record of a fast_math run names neither the numerics class nor fast_math, while ADR-008 and the acceptance numerics-class-follows-fast-math say the run record names the class tolerance
- file: DESIGN.md
  line: 622
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: undecided
  message: ADR-008 forbids reassociation but implements fast_math by dropping NoContraction, which also stops forbidding reassociation, and the tolerance test stayed green under a reassociated loss-sum mutant
```
