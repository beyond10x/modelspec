---
format: aep.planning-md/3
id: review-result:adversary-bert-embedding-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-embedding-lookup
relations:
- reviews: story:bert-embedding-lookup
revision: 1
---
```
unit: story:bert-embedding-lookup, worktree wt-dd1e45b09112 at head 81111b8 plus 2 untracked adversary test files
verdict: red
cases: executed 1288→1307, red 2
origin: introduced 2 / pre-existing 0 / undecided 0
wrote-outside-worktree: 3 paths (listed in part 6)
needs-coordinator: severity call on F1 and F2 (I rated both warning: the acceptance statement does not name `compile`)
```

**1. What I changed in the tree**

`git --no-pager diff --stat` is empty, because the tracked files are untouched. `git status --short` shows only two new files:
```
?? crates/modelspec-cli/tests/adversary_embedding_cli.rs
?? crates/modelspec-runtime/tests/adversary_embedding_runtime.rs
```
Both are test files, so the charter held. Neither file contains a `/home/` path.

**2. Cases added**

Two cases are red. I ran each alone before the suite, then marked it `#[ignore]` with its finding id:
- **F1, `adversary_embedding_compile_for_the_cpu_writes_an_artifact`.** It runs `modelspec compile <unit fixture embedding_ids.mspec> --backend cpu --output …` and expects exit 0 with an `artifact.json`. Red:
  ```
  assertion `left == right` failed: E-INTERNAL-002: compiler bug: the schedule refused the training step
    error: %3 is Embedding, which has no Vulkan kernel
    this is a fault in modelspec, not in the specification
    left: Some(1)
   right: Some(0)
  ```
- **F2, `adversary_embedding_no_kernel_is_not_a_compiler_bug`.** It checks that `ir --stage kernel`, `train --backend vulkan` and `infer --backend vulkan` do not answer with E-INTERNAL. Red on the first command with the same three lines. I ran the other two directly: same output, exit 1. All three stop at compile time, before any device is opened.

Green now: 6 CLI cases (plus a seventh that only asserts something when the compile is refused) and 13 runtime cases. Attacks 1, 2, 3, 4 and 6 are all covered by them.

**3. Suite run (after the cases existed)**

`b10x-build-slot cargo test --workspace --no-fail-fast` exited 0: 1307 passed, 0 failed, 111 ignored (my 2 included). The "before" figure, 1288, is 1307 minus my 19 executed cases. I did not run the `--features vulkan` suite.

**4. Findings**

- **F1** at `crates/modelspec-core/src/compile.rs:75`, NEEDS-CHANGE, introduced. `compile_artifact` runs the Vulkan schedule for every backend. So `compile --backend cpu` refuses every program that uses `embedding`, including the unit's own clean fixture, even though a CPU artifact has no kernels. What reaches it is the documented `compile --backend cpu` command. CPU `train` does not call `compile_artifact`. Suggested fix: skip the schedule for `Backend::Cpu`.
- **F2** at `crates/modelspec-core/src/compile.rs:90`, NEEDS-CHANGE, introduced. `ScheduleError::NoKernel` (added at `schedule.rs:786`) is not a target limit, so it is reported as E-INTERNAL-002. That code's documentation says "No source file is meant to reach it". The implementor's report says this only happens for `device = vulkan`, but it happens for any `device` setting. What reaches it is `ir --stage kernel`, `train --backend vulkan`, `infer --backend vulkan` and `compile --backend vulkan`. Suggested fix: a user-facing E-TARGET code for an operator with no kernel.

**5. Attacked and not broken**
- **Attack 1:** an `i32` in every other operand position of all 10 operators gives E-TYPE-018 at the call, one diagnostic each. Other misuses give existing codes:
  - an `i32` returned through a `let`: E-TYPE-007
  - an `i32` passed to an `f32` argument, or the reverse: E-TYPE-008
  - float ids: E-TYPE-018; integer-literal ids: E-TYPE-005
  - an `i32` target read by `mse`: E-TYPE-018 at the loss
- **Attack 2:** ids of rank 0 or 2 and a table of rank 1 or 3 give E-TYPE-004; `N = 0` gives E-TYPE-015. The verifier itself refuses an `i32` operand of 11 ops in hand-built IR, an `i32` output and a wrong result type.
- **Attack 3:** ids equal to `V`, `-1`, `i32::MIN` or `i32::MAX` give `IdOutOfRange` at the right position. Wrong length, an unknown name, or a tensor in the wrong map are refused. The range check runs before a node that would fail to allocate 2^50 elements. One plan reruns correctly with new ids, including after a refused run.
- **Attack 4:** over 64 seeded cases with values from 1e-8 to 1e8, the gradient matches a serial ascending f32 sum bit for bit on two runs, and unread rows are exactly `+0.0`. `V = 1, H = 1`, `matmul → embedding → matmul` and the input adjoint of a table input all pass.
- **Attack 5:** the five constants are byte-identical to the base. Parameters are perturbed as before; f32 inputs were never perturbed, at base either. The scope names `tests/common/mod.rs`, but the harness is at `common/finite_difference.rs`, which is harmless.
- **Attack 6:** E-DATA-004 fires for an unread `i32` input (ahead of E-DATA-003), for an `i32` target (ahead of E-DATA-002), and for dense data before any file is opened.
- **Attacks 7 and 9:** `docs-check`, `spec-validate`, both schema checks and `ui-types-check` exit 0. `TensorOperator` and `DType` are not in `ui/src/generated`, so I had nothing to plant there.
- **Attack 8:** a refused compile leaves no output directory.
- **Attack 10:** each of the five mutants ran on a copy in scratch, and each was killed by a test the unit wrote:

| Mutant | Killed by |
|---|---|
| descending scatter | `embedding_grad_accumulates_rows_in_ascending_n` |
| skip the range check | `embedding_id_out_of_range_is_a_runtime_error` |
| treat `-1` as `V - 1` | `embedding_id_out_of_range_is_a_runtime_error` |
| accept `i32` in `add` | `ops::tests::an_i32_operand_of_any_other_op_is_refused` |
| emit E-TYPE-017 instead of 018 | `types::tests::an_operand_of_the_wrong_dtype_is_e_type_018` |

**6. Paths written outside the worktree**

Under `<wave-root>/embedding/adv1-scratch/`: `diff-runtime-cli.txt`, `gradients-base.rs` and `suite.log`. The mutant copy's target was removed with `cargo clean --target-dir`, then the copy deleted. The CLI cases also write their fixtures under the worktree's own `target/tmp/adversary_embedding_cli/`; I left that alone.

```findings
- file: crates/modelspec-core/src/compile.rs
  line: 75
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: compile_artifact schedules for Vulkan regardless of backend, so `modelspec compile --backend cpu` refuses every embedding program, the unit's own clean fixture included, with E-INTERNAL-002.
- file: crates/modelspec-core/src/compile.rs
  line: 90
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: ScheduleError::NoKernel for Embedding is reported as E-INTERNAL-002 "a fault in modelspec" on ir --stage kernel, train/infer/compile --backend vulkan, though that code is documented as unreachable from any source file.
```
