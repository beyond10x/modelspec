---
format: aep.planning-md/3
id: review-result:adversary-m0-parse-subset-pass-1
kind: review-result
status: active
title: Adversary pass 1 on m0-parse-subset
relations:
- reviews: story:m0-parse-subset
revision: 1
---
unit: story:m0-parse-subset, uncommitted working tree on base 1c8f16b (worktree `modelspec-impl-m0-parse-subset`)
verdict: NEEDS-CHANGE
cases: executed 75→86 (syntax lanes; 75 is the count the implementor reported), red 3
origin: introduced 3 / pre-existing 0 / undecided 0
wrote-outside-worktree: 2 paths
needs-coordinator: whether a stack overflow on input nested about 2,000 levels deep blocks M0 or becomes a story

**1. Diff stat.** `git --no-pager diff --stat` lists only the implementor's 4 tracked `src/syntax/*` files. My only added path is the untracked `crates/modelspec-core/tests/syntax_adversary.rs`. I edited no implementation file. A release build of the test binary also wrote `target/release/` inside the worktree.

**2. Cases** in `crates/modelspec-core/tests/syntax_adversary.rs`. Each was run alone through `b10x-build-slot cargo test -p modelspec-core --test syntax_adversary -- --exact <name>`:

| Case | Asserts | Now | Exit |
|---|---|---|---|
| `unclosed_model_does_not_report_the_following_train_as_missing` (:118) | a model missing its `}` gives no E-SYNTAX-020 when a train follows | red | 101 |
| `unclosed_function_does_not_report_the_following_train_as_missing` (:134) | the same, through a function missing its `}` | red | 101 |
| `deeply_nested_calls_do_not_overflow_the_stack` (:153) | `f(f(...x...))` nested 10,000 deep returns Ok or Err, run in a child process on an 8 MiB thread | red | 101 |
| 8 others: every prefix and every one-character edit of `examples/linear_regression.mspec` (inserting é € 😀 BOM `{` `}` `<` `@` CR), multibyte E-001 at 2:12, extreme literals round-trip, named and nested args round-trip, identifiers that start with a keyword, comment at EOF, 20,000 inputs from a fixed-seed generator (no panic, every accepted one round-trips) | | green | 0 |

Red output, captured verbatim from the run-alone runs:
```
the file has a train item, yet: [("E-SYNTAX-002", 6, 1), ("E-SYNTAX-020", 16, 1)]
the file has a train item, yet: [("E-SYNTAX-002", 6, 1), ("E-SYNTAX-020", 16, 1)]
child exited ExitStatus(unix_wait_status(134)); stderr:
thread '<unknown>' (2450757) has overflowed its stack
fatal runtime error: stack overflow, aborting
```
To find the threshold I swept the depth on an 8 MiB thread. A debug build passes at 1,000 and aborts at 2,000. A release build passes at 4,000 and aborts at 6,000.

**3. Suite**, run after the cases existed: `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci` exited 201. Build and lint passed. `syntax`: 75 passed. `syntax_adversary`: 8 passed, 3 failed (the three above). `task test` stops at the first failing test binary, so the binaries after it did not run.

**4. Findings** (covering the working tree on 1c8f16b)

| file:line | Finding | Verdict | Origin | What reaches it |
|---|---|---|---|---|
| src/syntax/parser.rs:285 (`recover`) + :258 | After an error inside an item, recovery counts braces from the item's first `{`. An unclosed `{` therefore swallows the following `train` item, and the parser then emits a false "the file has no `train` item" (E-SYNTAX-020). This also breaks "first error per item". | NEEDS-CHANGE | introduced | Any file with a missing `}`, which is a common typo. Fix: stop recovery at a top-level `model` or `train` keyword at any depth, or suppress E-020 once recovery has skipped a `model`/`train` token. |
| src/syntax/parser.rs:685/720/729 (`expr`→`call`→`args`) | Unbounded recursion: a stack overflow aborts the process instead of returning Err. | CONFIRMED | introduced | Only a file nested about 2,000 levels deep, which in practice means generated or hostile input. No real model nests like that. Fix: a depth limit that returns a new E-SYNTAX code, or an existing one. |

**5. Attacked and not broken**
- Every char-boundary prefix and every one-character edit of linear_regression: no panic.
- Multibyte unrecognised characters: one E-001, column counted in chars.
- Round-trips: f64 extremes (5e-324, 1.79e308, 1e15/1e16, 0.30000000000000004, `1E5`, `2.5e+3`), u64::MAX seed and dimension, leading zeros, nested named args and member calls.
- Lexer: keyword-prefixed identifiers (`modelx`, `fnord`, `letter`, `returned`, `targets`), comment at EOF without a newline, CRLF and a comment containing é.
- Fixed-seed fuzz over 20,000 inputs: no panic, no round-trip loss.
- The implementor's suite already pins: tabs, CRLF, end-of-input position, the § 8.1 layout, the E-009 operators, number overflow.

Judgement note, not raised as a case: `module`, `layer`, `const` and `repeat` are accepted as identifiers (`let layer = x;` parses). That is legal for M0, but such files stop parsing once iteration 3 makes them keywords.

**6. Written outside the worktree**
- `~/.cache/modelspec-wave-m0/m0-parse-subset/scratch/adversary/` (the directory)
- `~/.cache/modelspec-wave-m0/m0-parse-subset/scratch/adversary/task-ci-pass1.log`

```findings
- file: crates/modelspec-core/src/syntax/parser.rs
  line: 285
  category: boundary
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: brace-counting recovery after an unclosed item swallows the following train item and emits a false E-SYNTAX-020 "file has no train item" (red in syntax_adversary.rs:118 and :134)
- file: crates/modelspec-core/src/syntax/parser.rs
  line: 685
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: unbounded recursion in expr/call/args aborts with a stack overflow on calls nested about 2000 deep (debug) or 6000 deep (release) instead of returning Err (red in syntax_adversary.rs:153)
```
