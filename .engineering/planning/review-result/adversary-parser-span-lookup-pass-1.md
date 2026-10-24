---
format: aep.planning-md/3
id: review-result:adversary-parser-span-lookup-pass-1
kind: review-result
status: active
title: Adversary pass 1 on parser-span-lookup-not-linear-in-column
relations:
- reviews: story:parser-span-lookup-not-linear-in-column
revision: 1
---
```
unit: story:parser-span-lookup-not-linear-in-column, uncommitted working tree on base 93121b3 (worktree modelspec-impl-span)
verdict: NEEDS-CHANGE (flaky timing assertion); columns: nothing found
cases: executed 349→354, red 0
origin: introduced 1 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 path (directory <wave-root>/span/scratch/adversary/)
needs-coordinator: none
```

**1. `git --no-pager diff --stat`** (tracked files):
```
 crates/modelspec-core/src/syntax/parser.rs | 69 ++++++++++++++++++++++++------
 1 file changed, 57 insertions(+), 12 deletions(-)
```
That line is the implementor's change. The only file I added is the untracked test `crates/modelspec-core/tests/span_lookup_adversary.rs`. I touched no implementation file.

**2. Cases I added** in `<trees>/modelspec-impl-span/crates/modelspec-core/tests/span_lookup_adversary.rs`. All 5 were green on their first run, alone (`test result: ok. 5 passed`). I wrote no red cases.

| Case | What it checks |
|---|---|
| `empty_and_blank_sources` | `""` gives 1:1, `"\r\n\r\n"` gives 3:1, `"\r\r"` gives 1:3 (a lone CR counts as a column), `"\t\t"` gives 1:3 |
| `crlf_lone_cr_and_tab_before_a_multi_byte_character` | CRLF, a lone CR and a tab before é and 漢 |
| `four_byte_combining_and_adjacent_unrecognised_characters` | two emoji back to back, an emoji glued to `param`, a combining mark, characters of 1 to 4 bytes then end of input, a BOM at 1:1 |
| `positions_after_error_recovery_across_line_ends` | positions after recovery, across CRLF, lone CR and LF |
| `nesting_too_deep_far_right_after_multi_byte_comment` | E-SYNTAX-021 at the 65th `(`, checked against a separate implementation of the base rule |

**3. Suite run, after my cases existed:** `b10x-build-slot cargo test -p modelspec-core` exited 0 with 354 passed, 0 failed. The before count is 354 minus my 5. `cargo fmt --check` and clippy (`-D warnings`) are clean on my file.

**4. Finding.** `crates/modelspec-core/tests/span_lookup.rs:66` (`assert!(ratio <= 1.2)`) fails when the machine is loaded. Each check takes about 88 ms in the debug build, and each side's median comes from only 3 runs measured one after the other.

| Load on this 20-core host | Runs | Failed | Ratio range |
|---|---|---|---|
| the test alone | 15 | 0 | 0.508–1.020 |
| 8 copies at once | 40 | 0 | 0.918–1.046 |
| 30 copies at once | 120 | 25 (exit 101) | 0.293–2.018 |

- **What reaches it:** this host runs 3 build slots, so `task ci` can run while other units compile on every core. A shared CI runner gives the same contention.
- **Order bias:** the one-per-line side is always measured first. In the first lone run it took 182 ms cold, against about 88 ms warm. Warm-up therefore lands on the baseline, which makes a regression look smaller than it is.
- **Fix, not applied:** alternate the two sides and take the minimum of N runs. Or assert on counted work rather than wall time, or move the timing to an `#[ignore]`d measurement.

**5. What I attacked and could not break:**
- **Columns:** the new code computes exactly what the old code did for any offset on a character boundary. CR, CRLF and tabs are unchanged.
- **Character boundaries:** every lexeme start falls on a character boundary for characters of 1 to 4 bytes (logos 0.16.1). So the new slice in `Parser::new` adds no panic.
- **Offsets `span_at` receives** (`parser.rs:121`): only lexeme starts, 0 and `source.len()`. An offset inside a character can't reach it.
- **Acceptance in release:** `modelspec check` on the CLI, 5 rounds of median-of-3. One line took 30–32 ms against 30–32 ms one per line, ratio 0.94–1.07. One earlier round under load gave 57 vs 86 ms, so the criterion itself is noisy at this size.

**6. Written outside the worktree:** the directory `<wave-root>/span/scratch/adversary/`, holding:
- scripts: `ratio.sh`, `gen.sh`, `time.sh`
- inputs: `one.mspec`, `per.mspec`
- logs: `alone.log`, `par8.log`, `par30.log`, `suite.log`

I ended my lease, then ran rustfmt on my own test file (import order only) and re-ran its 5 cases green.

```findings
- file: crates/modelspec-core/tests/span_lookup.rs
  line: 66
  category: flaky
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the debug-build ratio <= 1.2 assertion on a median of 3 sequential ~88 ms checks failed 25 of 120 runs with 30 copies at once on 20 cores (ratio 0.293-2.018), and always measuring the baseline first puts warm-up on the baseline
```
