---
format: aep.planning-md/3
id: review-result:m1-g-replay-adversary-1
kind: review-result
status: active
title: 'm1-g unit 5 adversary pass 1: 3 red (double record-failure line, refused serve deletes a recording, second RunStarted served)'
relations:
- reviews: story:observatory-run-recording-replay
revision: 1
---
```
unit: story:observatory-run-recording-replay, tree modelspec-impl-m1g-replay at 4309836 plus one untracked test file
verdict: CONFIRMED
cases: executed 931→935, red 3
origin: introduced 3 / pre-existing 0 / undecided 0
needs-coordinator: whether observe refuses event orders beyond the listed refusals (a second RunStarted); settled by the coordinator: refuse it
```

1. **Cases added** in `crates/modelspec-cli/tests/adversary_observatory_record.rs`:

| Test | On 4309836 |
|---|---|
| `adversary_a_record_write_failure_is_reported_on_one_line` | red: 2 "cannot record to /dev/full" lines, expected 1 |
| `adversary_a_refused_serve_leaves_an_existing_recording_alone` | red: the existing file at `--record` is gone after "Address already in use" |
| `adversary_observe_refuses_a_second_run_started` | red: RunStarted, RunStarted, RunFinished is served instead of refused |
| `adversary_a_divergent_run_records_and_replays` | green |

2. **Suite**: `cargo test --workspace --no-fail-fast` exit 101: 932 passed, 3 failed (only `adversary_observatory_record`), 49 ignored. The unit's own `observatory_record.rs` 6 passed, `modelspec-observatory/tests/record.rs` 3 passed. fmt, clippy `-D warnings`, `task docs-check`, `ess specify validate` (7 files, valid) and the schema `--check` (83 artifacts, current) all pass.

3. **Attacked, not broken**: an empty file, a lone blank line, a blank last line, a BOM, trailing garbage, invalid UTF-8, an unknown field, an unknown event kind, a directory and a missing file are each refused with exit 1 and one line; a non-loopback `--serve` exits 2; `--record` into a directory or a missing directory exits 1 with no stdout; an E-TRAIN-001 recording replays; two clients, one late, read the same stream; the recording equals the live stream byte for byte.

4. **Noted without a test**: a CRLF file is accepted and replayed without the `\r`; `--record` naming the source `.mspec` truncates it after it was read.

```findings
- file: crates/modelspec-cli/src/train.rs
  line: 497
  category: acceptance
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: record_failure takes record_error before emit(last), so the final event is written again after a failed write and a full disk prints two "cannot record" lines instead of the one the contract promises (adversary_a_record_write_failure_is_reported_on_one_line)
- file: crates/modelspec-cli/src/train.rs
  line: 412
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the --record file is created (truncating an existing one) before --serve binds, and a refused bind then deletes it, destroying a previous recording for a run that never started (adversary_a_refused_serve_leaves_an_existing_recording_alone)
- file: crates/modelspec-observatory/src/record.rs
  line: 146
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: parse_recording checks only that the first event is RunStarted, so a file with a second RunStarted, which RunEventBody's order forbids, is accepted and served (adversary_observe_refuses_a_second_run_started)
```
