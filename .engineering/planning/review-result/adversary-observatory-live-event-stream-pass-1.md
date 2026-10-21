---
format: aep.planning-md/3
id: review-result:adversary-observatory-live-event-stream-pass-1
kind: review-result
status: active
title: Adversary pass 1, story:observatory-live-event-stream
relations:
- reviews: story:observatory-live-event-stream
revision: 1
---
```
unit: story:observatory-live-event-stream
verdict: red
cases: executed 332→339, red 2
origin: introduced 4, pre-existing 0, undecided 0
wrote-outside-worktree: ~/.cache/modelspec-wave-observatory-1/observatory-live-event-stream/scratch/adv/ (logs, probe scripts, ess-* spec copies, ess-dl/ release tarball, base/ extract of ab9ed27 with its own target/, 211M)
needs-coordinator: yes. Finding 2 is a settlement made without a source, which the story says to report rather than choose. The 211M scratch build in adv/base/target is left for the coordinator, since target/ dirs are never deleted by hand.
```

The unit is red: two new cases fail, both because a training run whose loss goes non-finite streams events that fail the unit's own schema. The findings cover commit 99902f4 plus my two untracked test files.

**1. Diff stat.** `git diff --stat` is empty because nothing tracked changed. `git status --short` shows only my two test files, both untracked:
```
?? crates/modelspec-cli/tests/observatory_stream_adversary.rs
?? crates/modelspec-observatory/tests/adversary_schema.rs
```

**2. Cases added**

| Case | Asserts | Now |
|---|---|---|
| `adversary_schema::a_non_finite_loss_frame_is_schema_valid` | a `StepCompleted` frame with a NaN or ±inf loss validates against the RunEvent schema | red |
| `adversary_schema::the_schema_refuses_what_the_spec_forbids` | the validator rejects 12 planted invalid events (loss null or text, step -1, missing state, unknown kind, extra field, non-uuid id, and others), so it is not a validator that accepts anything | green |
| `observatory_stream_adversary::a_diverging_run_streams_only_schema_valid_events` | `train --serve` on the linear example with `lr = 1000.0`, 50 steps: every event validates | red |
| `observatory_stream_adversary::last_event_id_boundaries` | resume after id 0 starts at 1; after the last id, last+10 and u64::MAX the body is empty | green |
| `observatory_stream_adversary::concurrent_and_disconnecting_clients` | a client that drops mid-stream, then two concurrent clients and a late one, all read the same gap-free stream ending in `RunFinished` | green |
| `observatory_stream_adversary::a_port_in_use_exits_1_without_running` | exit 1, empty stdout, `cannot serve on` on stderr | green |
| `observatory_stream_adversary::an_unparsable_serve_address_is_a_usage_error` | ports 99999, empty and -1 exit 2 | green |

Red output from each case run alone, before any suite run (logs `red-nonfinite.log` and `red-diverging.log`, both EXIT=101):
```
loss NaN: {"run_event_id":"8c4f…","sequence":0,"body":{"kind":"StepCompleted","value":{"step":0,"loss":null,"metrics":[],"parameters":[]}},"state":"Emitted"}
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.01s
```
```
45 of 53 events fail the schema, first: ["7: {\"kind\":\"StepCompleted\",\"value\":{\"loss\":null,…,\"step\":6}}", …]; stdout ends ["param @LinearReg.w: Tensor<f32, [4, 1]> = [NaN, NaN, NaN, NaN]", "step 50 loss NaN"]
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out; finished in 0.04s
```

**3. Suite runs (after the cases existed)**

| Command | Summary | Exit |
|---|---|---|
| `cargo test -p modelspec-observatory -p modelspec-runtime -p modelspec-cli --no-fail-fast` | 337 passed, 2 failed (the two red cases above) | 101 |
| the same with `--skip` on my 7 case names | 332 passed, 0 failed, 7 filtered | 0 |
| `cargo clippy … --all-targets -- -D warnings` (after fixing two lints in my own test files) | `Finished` | 0 |
| `cargo fmt -p … -- --check` | no output | 0 |
| final re-run of the full suite after the lint fixes | 337 passed, 2 failed | 101 |

**4. Findings**

| # | file:line | What was measured | What reaches it | Verdict | Origin |
|---|---|---|---|---|---|
| 1 | `crates/modelspec-observatory/src/server.rs:603` | `serde_json` writes a non-finite f64 loss as `null`. The schema requires a number, so 45 of 53 events of a diverging run are invalid, and the last loss cannot match the CLI's `step 50 loss NaN`. | Any valid `.mspec` with a large `lr`. Plain `train` exits 0 on it at both base and HEAD. The stream is new in this unit. | NEEDS-CHANGE | introduced |
| 2 | `spec/domains/observatory.yaml` (StreamFrame summary) | "Keeps serving until interrupted" names no source; it cites the story's own open question. Probe: an E-DATA-003 run and a finished run were both still running after 10 s, and SIGTERM gave exit 143. Under `--serve` the exit status never reports 0 or 1. | Every `--serve` invocation. AGENTS.md defines exit codes 0 and 1. | NEEDS-CHANGE | introduced |
| 3 | story Acceptance, scenario 5 | `observatory.stream.schema-current` is the `task schema-check` step, not a named test. Scenario ids are mapped only in a doc comment, not used as test names. | `task ci` | CONFIRMED | introduced |
| 4 | `AGENTS.md` (`task schema-check` line) | It says the check "fails when schema/ differs". A hand-added `stale.schema.json` passes the check (exit 0). A type deleted from the spec has its file removed by `schema-generate`, so regeneration produces no orphan. | Only a hand-placed file; nothing found that produces one. | INFEASIBLE | introduced |

Fix for finding 1, not applied: refuse or report a non-finite loss before emitting, for example as `RunFailed`, or make the spec say how a non-finite loss is carried. Finding 2 needs a source to settle the question, or the process should exit with the run's code once the stream has ended.

**5. Attacked and not broken**
- Plain `train` without `--serve`, base binary against HEAD binary on 3 inputs (linear example, diverging, E-DATA-003): stdout and stderr byte-identical, exit codes the same (0, 0, 1).
- Last-Event-ID at 0, at the last id, past the last id, at u64::MAX: correct. Non-numeric (`x`) restarts from 0, as the unit's own test already covers.
- Concurrent clients, a client that disconnects mid-stream, a late client: no hang, no gap, ids monotonic.
- A non-loopback address exits 2, a bad port exits 2, a port in use exits 1.
- `schema-check` in scratch copies: a planted schema edit fails (exit 1), a deleted schema file fails (exit 1), a spec summary edit fails (exit 1). `spec-validate` fails on a planted bad type reference (exit 1, `domains/observatory.yaml:86:5`).
- check.yml: the downloaded 0.66.0 tarball's sha256 equals the pinned value. The download works without auth (HTTP 200, and `beyond10x/ess` is public). The tarball contains `ess-0.66.0-…/ess`. A one-character hash mismatch fails the step with exit 1 before `tar` runs.
- No home paths in `.ess-output/state.json`.

**6. Paths written outside the worktree:** everything is under `~/.cache/modelspec-wave-observatory-1/observatory-live-event-stream/scratch/adv/`:
- **Logs:** `red-*.log`, `suite-*.log`, `clippy.log`, `ess-*.log`, `orphan-*.log`, `base-build.log`
- **Scripts:** `ess-probe.sh`, `cli-probe.sh`, `sha-probe.sh`, `orphan-probe.sh`
- **Spec copies:** `ess-{clean,edit,extra,missing,specdefect,specsummary,orphan}/`
- **Download:** `ess-dl/`
- **Base build:** `base/`, including `base/target/` at 174M
- **Probe outputs:** `diverging.mspec`, `unread_x.mspec`, `base.out`, `base.err`, `head.out`, `head.err`, `serve.out`, `serve.err`

Every server process I started was killed by its PID; `pgrep -x modelspec` now shows none. Free space on `/` is 23G.

```findings
- file: crates/modelspec-observatory/src/server.rs
  line: 603
  category: contract-drift
  severity: blocker
  verdict: NEEDS-CHANGE
  origin: introduced
  message: a non-finite f32 loss is framed as "loss":null, so a diverging run (lr = 1000.0) streams 45 of 53 events that fail the committed RunEvent schema StreamFrame.data promises they satisfy
- file: spec/domains/observatory.yaml
  category: judgement
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the keeps-serving-until-interrupted settlement cites no source as the story requires, and makes train --serve never exit 0 or 1 (measured exit 143 on SIGTERM for both a failed and a finished run)
- file: .engineering story:observatory-live-event-stream Acceptance scenario 5
  category: acceptance
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: scenario observatory.stream.schema-current is the task schema-check step rather than a named test, and no test carries a scenario id as its name
- file: AGENTS.md
  category: contract-drift
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: task schema-check passes with a hand-added stale schema file although the new AGENTS.md line says it fails when schema/ differs; regeneration produces no orphans, so nothing found reaches it
```
