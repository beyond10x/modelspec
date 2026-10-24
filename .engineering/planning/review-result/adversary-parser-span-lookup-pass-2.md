---
format: aep.planning-md/3
id: review-result:adversary-parser-span-lookup-pass-2
kind: review-result
status: active
title: Adversary pass 2 on parser-span-lookup-not-linear-in-column
relations:
- reviews: story:parser-span-lookup-not-linear-in-column
revision: 1
---
```
unit: story:parser-span-lookup-not-linear-in-column, pass 2, uncommitted working tree on base 93121b3 (parser.rs as of 10:31:34)
verdict: CONFIRMED (the gated scan counter cannot fail for an uncounted lookup walk)
cases: executed 354→355, red 0
origin: introduced 2 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 path (directory <wave-root>/span/scratch/adversary/)
needs-coordinator: none
```

**1. `git --no-pager diff --stat`** (tracked files):
```
 crates/modelspec-core/src/syntax/parser.rs | 92 +++++++++++++++++++++++++-----
 1 file changed, 79 insertions(+), 13 deletions(-)
```
That line is the implementor's change. I added one untracked test, `crates/modelspec-core/tests/span_lookup_counter_adversary.rs`, next to my pass-1 file `span_lookup_adversary.rs`. I touched no implementation file.

**2. Case added.** `scan_count_is_the_anchor_pass_alone` checks that `span_lookup_scan_count` equals `source.chars().count()` for 6 inputs: 2 000 lets on one line and one per line, each with and without a diagnostic far to the right, an error-recovery input with CRLF, and the empty source. It was green on its first run alone (`1 passed`). That green result is the evidence: every lookup the parser makes lands exactly on an anchor, so lookups add 0 to the count. The count is only the one pass in `Parser::new`.

**3. Suite, after the case existed:**
- `b10x-build-slot cargo test -p modelspec-core` exited 0: 355 passed, 0 failed, 1 ignored.
- The three span test files run together are green: 4 passed + 1 ignored, then 5, then 1.
- `cargo fmt --check` exited 0, clippy `--all-targets -D warnings` is clean, and `task docs-check` reports its files current.

**4. Findings**
- **The counter can undercount.** `crates/modelspec-core/tests/span_lookup.rs:67` reduces to "the one-line source has at most 1.2x the characters of the one-per-line source". The test already pins the two lengths equal at `:54`. The counter is incremented only inside `Anchor::advance` (`parser.rs:95`).
  - A lookup that walks without calling `advance` adds nothing. Example: putting back the base `source[line_start..offset].chars().count()` in `span_at` (`parser.rs:144`). The count stays at the anchor pass, so the gate stays green on the regressed algorithm.
  - It does catch anchors placed too sparsely (walks through `advance`).
  - I did not compile that mutant: the rules forbid copying the tree. The conclusion follows from the case in part 2 together with the single increment site.
  - No gated test now covers the acceptance figure (wall time of `modelspec check`); it is only an `#[ignore]`d measurement.
  - Fix (not applied): count in `span_at` itself, independently of `advance`. Or assert that every lookup offset hits an anchor exactly.
- **New public API item.** `modelspec_core::syntax::parser::span_lookup_scan_count` (`parser.rs:51`) exists only for a test. The base `parser` module exposed nothing public (`parse` is `pub(crate)`). It needs a CHANGELOG line or `#[doc(hidden)]`, which is the coordinator's call.

**5. Attacked and could not break**
- **`file(&mut self)`:** the type is private and `parse` is unchanged, so behaviour is the same. The full suite is green.
- **Cost on the production path:** release `modelspec check` on 20 000 lets, 5 rounds of median-of-3:

  | Layout | Pass 1 | Pass 2 (with the counter) |
  |---|---|---|
  | one per line | 30–32 ms | 30–34 ms |
  | one line | 30–32 ms | 31–33 ms |

  The release binary (10:33:20) is newer than parser.rs (10:31:34), so it includes the counter.
- **Both untracked test files together, plus mine:** green.

**6. Written outside the worktree:** the directory `<wave-root>/span/scratch/adversary/`. It holds the pass-1 files plus `suite2.log`.

```findings
- file: crates/modelspec-core/tests/span_lookup.rs
  line: 67
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the scan count equals the anchor pass for every input because each lookup lands exactly on an anchor, and only Anchor::advance increments the counter, so an uncounted per-column walk in span_at (the base algorithm) leaves the gate green
- file: crates/modelspec-core/src/syntax/parser.rs
  line: 51
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: span_lookup_scan_count adds a test-only public item to modelspec_core::syntax::parser, which exposed nothing public at the base, and no CHANGELOG line or doc(hidden) covers it
```
