---
format: aep.planning-md/3
id: review-result:adversary-m0-check-types-shapes-pass-1
kind: review-result
status: active
title: Adversary pass 1 on m0-check-types-shapes
relations:
- reviews: story:m0-check-types-shapes
revision: 1
---
unit: story:m0-check-types-shapes, uncommitted working tree of ~/.local/state/worktree/trees/b10x/modelspec/modelspec-impl-m0-check-types-shapes (base a99317d)
verdict: NEEDS-CHANGE
cases: executed 289→301, red 5
origin: introduced 11 / pre-existing 1 / undecided 0
wrote-outside-worktree: 1 directory tree, ~/.cache/modelspec-wave-m0/m0-check-types-shapes/scratch/adversary/ (its copy/target/ is 242M)
needs-coordinator: none

**1. What I touched**

`git --no-pager diff --stat` lists only the implementor's 6 tracked files. My one file is untracked: `git status --short` shows `?? crates/modelspec-cli/tests/check_adversary.rs`. It is a test file, and I changed no other path in the tree.
- I ran `rustfmt --edition 2024` on that file only, after `cargo fmt --check` refused it.
- The CLI cases write files to `target/tmp/check_adversary/`, inside the tree's own build directory.
- I put no fixture directory under `tests/language/check/adversary/`. The implementor's `every_fixture_file_is_a_test_case` reads directory entries, so that subdirectory alone would have turned its test red. My sources are inline instead.

**2. Cases added** (in `crates/modelspec-cli/tests/check_adversary.rs`, 13 of them)

These 5 are red now. Each was run alone, and each output was captured before the suite ran:

| Case | Red output, verbatim (exit 101 each) |
|---|---|
| `function_named_like_earlier_parameter_reports_only_the_duplicate` | `left: [("E-NAME-003", 4, 8), ("E-NAME-008", 13, 22)]` `right: [("E-NAME-003", 4, 8)]` |
| `parameter_named_like_earlier_function_reports_only_the_duplicate` | `left: [("E-NAME-003", 6, 11), ("E-NAME-001", 9, 26)]` `right: [("E-NAME-003", 6, 11)]` |
| `wrong_initializer_does_not_hide_an_independent_matmul_mismatch` | `left: [("E-TYPE-013", 2, 36)]` `right: [("E-TYPE-013", 2, 36), ("E-TYPE-012", 5, 16)]` |
| `syntax_diagnostics_printed_in_source_order` | `E-SYNTAX-002 … :12:12` is printed before `E-SYNTAX-019 … :12:5`; `left: [(12, 12), (12, 5)]` `right: [(12, 5), (12, 12)]` |
| `edited_linear_regression_never_panics_or_reaches_internal` | `2 of 17759 edited sources broke a property; first: diagnostics out of source order: [(12, 12), (12, 5)]`. Same root as the row above (`model  {  LinearReg;`). No panic, no E-INTERNAL-001, no duplicate diagnostic. |

These 8 are green now. Each one kills a mutant that the implementor's suite lets live (section 5).
- `byte_count_overflow_without_element_overflow_is_e_type_016`
- `diagnostics_sorted_when_train_block_precedes_model`
- `xavier_uniform_on_rank_3_and_rank_0_is_e_type_014`
- `let_used_before_definition_or_by_itself_is_e_name_001`
- `vulkan_device_index_reaches_hir_train`
- `let_uses_resolve_to_their_own_binding_in_hir`
- `model_forward_resolves_when_forward_is_not_the_first_function`
- `each_source_operator_is_its_own_op_in_hir`

**3. Suite runs** (after the cases existed)

- `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci` exited 201. Build, fmt and clippy passed. The test task stopped at the first red binary: `tests/check.rs` `49 passed; 0 failed`; `tests/check_adversary.rs` `7 passed; 5 failed`. Because of that stop, `check-examples` did not run inside `ci`.
- The same command with `cargo test --workspace --no-fail-fast` exited 101: 301 executed, 296 passed, 5 failed. The 5 are the red cases above.
- Run on its own, `task check-examples` exited 0.

**4. Findings** (they cover the working tree on base a99317d)

| file:line | Verdict | Origin | What was measured | What reaches it |
|---|---|---|---|---|
| crates/modelspec-core/src/resolve.rs:132 | NEEDS-CHANGE | introduced | When a parameter and a function share a name, only the first is registered. Later uses then get a cascade E-NAME-008 or E-NAME-001. This contradicts the module doc: "uses after it are not reported again". | Any file declaring `param x` and `fn x`. Fix: register the second item in its own kind's map, or suppress lookups of names in `declared`. |
| crates/modelspec-core/src/types.rs:86 | NEEDS-CHANGE | introduced | `check_source` returns the parser's diagnostics without sorting them. The check.rs doc promises source order. | `modelspec check` on any file where parser recovery produces more than one error. Fix: sort in `check_source` as well. |
| crates/modelspec-core/src/types.rs:244 | CONFIRMED | introduced | A wrong initializer drops the parameter's valid declared type (line 153), so an independent E-TYPE-012 is hidden. The module doc says it "reports every diagnostic of the file". | Any non-`xavier_uniform` initializer plus a shape error on that parameter. |
| crates/modelspec-core/src/types.rs:98 | CONFIRMED | introduced | Deleting the sort keeps the implementor's suite green (mutant M1). | The only multi-diagnostic fixture is already emitted in source order. |
| crates/modelspec-core/src/types.rs:220 | CONFIRMED | introduced | Dropping the byte-count check keeps the suite green (M2). | Both E-TYPE-016 fixtures overflow the element count. |
| crates/modelspec-core/src/types.rs:247 | CONFIRMED | introduced | Changing `!= 2` to `< 2` keeps the suite green (M4): rank 3 is accepted. | The E-TYPE-014 fixture is rank 1 only. |
| crates/modelspec-core/src/types.rs:294 | CONFIRMED | introduced | Changing `LetRef(i)` to `LetRef(0)` keeps the suite green (M6). The HIR then points at the wrong `let`. | Lowering reads these refs. |
| crates/modelspec-core/src/types.rs:423 | CONFIRMED | introduced | Changing `Vulkan(index)` to `Vulkan(0)` keeps the suite green (M3). | `HirTrain.device`, which later stories read. |
| crates/modelspec-core/src/resolve.rs:141 | CONFIRMED | introduced | Changing `FnRef(functions)` to `FnRef(0)` keeps the suite green (M5). | Every fixture declares `forward` first. |
| crates/modelspec-core/src/resolve.rs:63 | CONFIRMED | introduced | Mapping `add` to `Sub` (M7) or `mean` to `ReduceSum` (M8) keeps the suite green. The ops type alike, and the unit test pins names only. | Lowering would emit the wrong op. |
| crates/modelspec-cli/tests/exit_codes.rs:42 | CONFIRMED | introduced | `unimplemented_check_exits_1` now passes only because `model.mspec` does not exist. Its name no longer says what it tests. | Nothing at runtime; it misleads readers. |
| crates/modelspec-core/src/syntax/parser.rs:282 | CONFIRMED | pre-existing | Recovery after `model  {` reports a spurious E-SYNTAX-019 "second `model` item". | parser.rs is byte-identical to a99317d (absent from the diff stat). |

**5. What I attacked and could not break**

- **E-INTERNAL-001:** not reached by any of 17759 truncations and token edits. By reading: no source dtype other than f32, `Input`/`ParamRead` are not in the operator table, and every `None` path carries a diagnostic.
- **Panics:** none in the checker across those edits. Separately, `modelspec check … | head -3` exits 101: `eprintln!` panics on a broken pipe. That affects the whole CLI, so I am not raising it as a finding.
- **Name resolution:** shadowing (E-NAME-004), use before definition, self-reference, duplicate data, unknown model/receiver/method, and parameters not visible in the train block.
- **Shape rules:** matmul inner dimension, scalar broadcast only, reductions to rank 0, `mse` requiring identical shapes, transpose rank, and the result-count overflow.
- **Train block:** zero `lr`, underflow and overflow; `steps = 0`; seed above u64, `adam` and `lr = 1e400` (all three refused by the parser).
- **Acceptance:** the matmul_mismatch stderr matches § 8.1 line for line; linear_regression exits 0 with no output; tiny_mlp exits 1 (E-SYNTAX-004/006/007); a directory argument exits 1.
- **Implementor's mutants killed:** M9 (`lr >= 0`) and M10 (loss-rank check removed) are caught by its fixtures.
- **Out of this unit:** "exactly one input and one target of the model output type" belongs to story m0-cpu-sgd-training (its `data.rs`), so I did not raise it here.

**6. Paths written outside the worktree**

- ~/.cache/modelspec-wave-m0/m0-check-types-shapes/scratch/adversary/: `copy/` (copy of the tree, with a 242M `target/`), `pristine/`, `probe/out_of_order.mspec`, `mutate.sh`, `adversary-names.txt`, `ci-1.log`, `ci-2.log`, `test-nff.log`, `mutants.log`, `mutant-M0…M10.log`, `tiny_mlp.err`. The whole directory is 1.6M without `copy/target/`.
- The harness wrote the background-task output file under ~/.cache/claude-tmp/claude-1000/.
- The worktree session lease `adversary-m0-check-types-shapes-p1` was acquired and then released.

```findings
- file: crates/modelspec-core/src/resolve.rs
  line: 132
  category: boundary
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: a parameter and a function sharing a name register only the first, so later uses add a cascade E-NAME-008 or E-NAME-001 against the module doc's no-re-report promise
- file: crates/modelspec-core/src/types.rs
  line: 86
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: check_source returns parser diagnostics unsorted, so modelspec check prints 12:12 before 12:5 despite check.rs promising source order
- file: crates/modelspec-core/src/types.rs
  line: 244
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: a wrong initializer discards the parameter's valid declared type and hides an independent E-TYPE-012 on it, against the every-diagnostic claim
- file: crates/modelspec-core/src/types.rs
  line: 98
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: deleting the source-order sort leaves the implementor suite green
- file: crates/modelspec-core/src/types.rs
  line: 220
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: dropping the byte-count overflow check leaves the implementor suite green
- file: crates/modelspec-core/src/types.rs
  line: 247
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: relaxing the xavier rank check to rank < 2 leaves the implementor suite green
- file: crates/modelspec-core/src/types.rs
  line: 294
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: binding every let as LetRef(0) leaves the implementor suite green while the HIR points at the wrong binding
- file: crates/modelspec-core/src/types.rs
  line: 423
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: mapping vulkan(N) to Vulkan(0) in HirTrain leaves the implementor suite green
- file: crates/modelspec-core/src/resolve.rs
  line: 141
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: numbering every function FnRef(0) leaves the implementor suite green because every fixture declares forward first
- file: crates/modelspec-core/src/resolve.rs
  line: 63
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: mapping add to Sub or mean to ReduceSum leaves the implementor suite green because the unit test pins operator names only
- file: crates/modelspec-cli/tests/exit_codes.rs
  line: 42
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: unimplemented_check_exits_1 now passes only because model.mspec is missing, so its name no longer describes what it tests
- file: crates/modelspec-core/src/syntax/parser.rs
  line: 282
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: pre-existing
  message: parser recovery after a malformed model setting reports a spurious E-SYNTAX-019 second model item
```
