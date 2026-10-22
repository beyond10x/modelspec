---
format: aep.planning-md/3
id: review-result:adversary-observatory-live-event-stream-pass-2
kind: review-result
status: active
title: Adversary pass 2, story:observatory-live-event-stream
relations:
- reviews: story:observatory-live-event-stream
revision: 1
---
unit: story:observatory-live-event-stream, head 627c650 plus 2 untracked test files in ~/.local/state/worktree/trees/b10x/modelspec/modelspec-impl-les-r2
verdict: NEEDS-CHANGE
cases: executed 343→348, red 1
origin: introduced 3 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 path, ~/.cache/modelspec-wave-observatory-1/observatory-live-event-stream/scratch-adv2/
needs-coordinator: none

The unit is red on one case. The loss summary in the fix's own spec text promises a stdout line naming NaN or ±inf for every null loss. Stdout prints only every 50th step and the last, so 44 of the 45 null-loss steps in a diverging run have no such line.

**1. Diff stat.** `git diff --stat` is empty because nothing tracked changed. `git status --short` shows only my two test files:
```
?? crates/modelspec-cli/tests/observatory_signal_adversary.rs
?? crates/modelspec-observatory/tests/adversary_floats.rs
```

**2. Cases added.** Each was run alone first (`red-floats.log` EXIT=0, `red-signal.log` EXIT=101).

| Case | Now | Summary |
|---|---|---|
| `adversary_floats.rs::every_float_round_trips_bit_exact_or_is_null` | green | 20,015 events with fixed-seed random bits plus edge values (-0.0, subnormals, MIN_POSITIVE, MAX, MIN, NaN payloads, ±inf) in loss, metric, norm and bin edges: a finite value passes the schema and reads back with the same bits, a non-finite one is null and reads back as NaN |
| `adversary_floats.rs::null_is_admitted_only_where_the_spec_puts_it` | green | null is accepted in the 4 float fields. The schema and `RunEvent` both refuse a missing loss, metric value or norm, a text bin edge, and null in step, sequence, counts, bin_edges, a metric name or metrics |
| `observatory_signal_adversary.rs::every_null_loss_has_the_stdout_line_that_names_it` | **red** | `44 of 45 null-loss steps have no `step <k> loss NaN\|inf\|-inf` line on stdout: [6, 7, …, 49]; the step lines are ["step 0 loss 9.693205", "step 50 loss NaN"]` |
| `observatory_signal_adversary.rs::two_signals_after_the_run_still_exit_with_the_runs_status` | green | TERM+TERM, INT+TERM and TERM+INT: a finished run exits 0, an E-DATA-003 run exits 1 |
| `observatory_signal_adversary.rs::a_connected_client_does_not_hold_the_exit` | green | a silent connection and one with a half-sent request head are open when SIGTERM arrives; exit 0 within 3 s |

**3. Gate (run after the cases existed)**

| Command | Summary | Exit |
|---|---|---|
| `cargo test -p modelspec-observatory -p modelspec-runtime -p modelspec-cli --no-fail-fast` | 347 passed, 1 failed (`observatory_signal_adversary`) | 101 |
| the same with my 5 cases skipped | 343 passed, 0 failed, 5 filtered | 0 |
| `cargo clippy … --all-targets -- -D warnings` | `Finished` | 0 |
| `cargo fmt -p … -- --check` (after a rustfmt of my own file) | no output | 0 |
| `task docs-check` | `website/docs: no raw admonition titles` | 0 |
| `ess specify validate --path .` (from `spec/`) | `modelspec v1 — 7 file(s), valid` (2 ESS-ENTITY-019 warnings, on lines the fix did not touch) | 0 |

**4. Findings**

| # | file:line | What was measured | What reaches it | Verdict | Origin |
|---|---|---|---|---|---|
| 1 | `spec/domains/observatory.yaml:79` | The `StepCompleted.loss` summary says the stdout line "for that step names which". `LOSS_EVERY = 50` (`train.rs:52`), so 44 of 45 null losses have no stdout line, and NaN and ±inf cannot be told apart for those steps. | Any diverging `.mspec`, for example `lr = 1000.0` with 50 steps. | NEEDS-CHANGE | introduced |
| 2 | `crates/modelspec-cli/src/train.rs:143` | The handler is installed after `train()` has written and flushed stdout. A SIGTERM sent on reading the last stdout line gave 143 in 5 of 300 runs and 0 in 295 (`signal-probe.log`). The spec text "after the run ends" does not say whether the end is the last stdout line or `RunFinished`. | A script that signals when it sees stdout complete. A client that waits for `RunFinished` is race-free, because the handler is installed before that event. | CONFIRMED | introduced |
| 3 | `spec/domains/observatory.yaml:147` | After the run, SIGHUP exits 0 (measured). During a plain run it exits 129. The spec and `--help` name only SIGINT and SIGTERM; only the `train.rs:158` doc comment mentions SIGHUP. | A terminal closing on a finished `--serve`. | CONFIRMED | introduced |

Fix for finding 1, not applied: reword the summary so the stdout line names the value only for the steps it prints (step 0, every 50th, and the last). This leaves the null decision as it is. For finding 2, the spec could define the run's end as the emission of `RunFinished` or `RunFailed`. For finding 3, the spec could name SIGHUP.

**5. Attacked and not broken**
- -0.0, subnormals, f32 and f64 MAX and MIN, NaN payloads: round-trip bit-exact. A test build always has serde_json `float_roundtrip` on, because jsonschema enables it, so a reader in a production build is not covered by this case.
- The schema admits no null outside the 4 floats. `null_when_absent` keeps the field required in both the schema and serde.
- The changed assertion in `the_schema_refuses_what_the_spec_forbids`: only "loss null" moved from refused to valid, as decided; the other 11 refusals are unchanged.
- `modelspec-docs` `cli.rs` test: changing the required marker to always "no" or always "yes" fails it (the check/ir Required-column assertion, or the `--serve | no` row). This was worked out from the code; I did not run the mutants. It is no weaker than before.
- Two signals, and clients still connected at the signal: the exit code is correct and the process does not hang.

**6. Paths written outside the worktree.** Everything is under `~/.cache/modelspec-wave-observatory-1/observatory-live-event-stream/scratch-adv2/`:
- **Logs:** `red-floats.log`, `red-signal.log`, `suite.log`, `suite-deselected.log`, `postfmt.log`, `clippy.log`, `fmt.log`, `docs-check.log`, `ess.log`, `signal-probe.log`
- **Probe script:** `signal-probe.sh`
- **Probe outputs:** `hup.body`, `hup.err`, `hup.out`

No modelspec process is left running (`pgrep -x modelspec` exits 1). My session lease is released.

```findings
- file: spec/domains/observatory.yaml
  line: 79
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the StepCompleted.loss summary promises a stdout line naming NaN, inf or -inf for each null loss, but the CLI prints only every 50th and the last step, so 44 of 45 null losses of a diverging run have none
- file: crates/modelspec-cli/src/train.rs
  line: 143
  category: concurrency
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the signal handler is installed after stdout is complete, so a SIGTERM sent on the last stdout line exits 143 instead of 0 in 5 of 300 runs
- file: spec/domains/observatory.yaml
  line: 147
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: after the run SIGHUP exits 0 with the run's status, which the spec and --help do not state (they name only SIGINT and SIGTERM; a plain run exits 129 on SIGHUP)
```
