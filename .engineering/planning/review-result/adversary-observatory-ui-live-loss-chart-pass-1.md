---
format: aep.planning-md/3
id: review-result:adversary-observatory-ui-live-loss-chart-pass-1
kind: review-result
status: active
title: 'Adversary pass 1: unbounded UI read, page stuck on a restarted server''s dead run'
relations:
- reviews: story:observatory-ui-live-loss-chart
revision: 1
---
# Adversary pass 1 on story:observatory-ui-live-loss-chart

Tree at 5b5ff34 plus 4 adversary test files. Verdict: two warnings, six notes, no blocker. 9 cases
added (executed 34 to 43), 5 red, no implementation file touched.

Ran: `cargo test -p modelspec-observatory --no-fail-fast` (unchanged suites pass: lib 3,
adversary_floats 2, adversary_schema 2, record 3, stream 17, ui_serve 8, ui_types 4; adversary files
ui_memory 0/1, ui_serve_adversary 3/4); `cargo test -p modelspec-cli --test observatory_ui --test
observatory_ui_adversary -- --include-ignored --test-threads=1` (observatory_ui 11/11, adversary
1/2); `pnpm typecheck` 0, `pnpm test` 11 of 13; clippy `-D warnings` and `cargo fmt --check` 0.
Not run: `task ci`, `task lint`, `build-without-node.sh`.

Red cases, verbatim:

- `a_file_over_the_limit_is_refused_without_reading_it_whole`: "Ui::read held 511 MiB at its peak to
  refuse a directory over the 64 MiB limit (one 512 MiB file)".
- `a_file_whose_name_a_browser_percent_encodes_is_served`: 404 where 200 was expected.
- `observatory_ui_follows_the_run_a_restarted_server_serves`: 30 s after a restart on the same
  address with a new run, the page still shows the old run, streaming and interrupted.
- `a series of finite values far apart draws a line with no NaN in it`: `M64,192 L624,NaN`.
- `the charts of a run have distinct names when a metric is called loss`: `["loss","loss"]`.

Attacked without a break: path traversal, `/events` precedence, symlinks and FIFOs, read before
bind, `--ui` without `--serve`, HEAD 405, ts-rs types against serde JSON, NaN and ±inf as gaps, no
reconnect after a finished stream, offline build, CSP against the app.

```findings
- file: crates/modelspec-observatory/src/ui.rs
  line: 76
  category: boundary
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: Ui::read reads each file whole before the 64 MiB check, so refusing one 512 MiB file peaked at 511 MiB resident.
- file: ui/src/lib/series.ts
  line: 58
  category: concurrency
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: after the server at the same address restarts with another run, apply skips its events and the page shows the dead run as streaming indefinitely.
- file: crates/modelspec-cli/tests/observatory_ui.rs
  line: 617
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the live scenarios' runs end within 0.07 s, before headless Chromium is ready (0.35-0.42 s), so they read an ended stream like a replay.
- file: crates/modelspec-observatory/src/ui.rs
  line: 16
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: no test reached the 4096-file or 64 MiB limits, so any value kept the suite green.
- file: crates/modelspec-observatory/src/server.rs
  line: 250
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: no test read the Content-Security-Policy header, so dropping it kept the suite green.
- file: crates/modelspec-observatory/src/ui.rs
  line: 98
  category: contract-drift
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: UI paths are matched without percent-decoding, so a file whose name a browser encodes is read but answered 404; no current build output has such a name.
- file: ui/src/lib/chart.ts
  line: 64
  category: property
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: finite values whose range overflows f64 put NaN in the chart path; no non-test metric producer exists and the f32 loss cannot reach it.
- file: ui/src/lib/series.ts
  line: 98
  category: property
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a metric named loss gives two charts with the same Vue key; no producer emits such a metric.
```
