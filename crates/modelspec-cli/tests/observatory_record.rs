//! `modelspec train --record <path>` and `modelspec observe <path>`: the recorded run file and its
//! replay over the event stream (story:observatory-run-recording-replay;
//! `spec/domains/observatory.yaml`, `RecordedRunFile`, `StreamFrame`).
//!
//! The recorded file is JSON Lines, one `RunEvent` per line in `sequence` order, each line the
//! JSON of a frame's `data:`. Every event read here is checked against the JSON Schema
//! `ess generate --kind schema` wrote for `modelspec.observatory.RunEvent`; nothing here reads the
//! Rust types of the stream.
//!
//! Scenario ids of the story map to the tests: `observatory.record.matches-live` is
//! [`observatory_record_matches_live`], `observatory.replay.matches-recording` is
//! [`observatory_replay_matches_recording`], `observatory.replay.refuses-damaged-file` is
//! [`observatory_replay_refuses_damaged_file`] and `observatory.record.fixture-current` is
//! [`observatory_record_fixture_current`].
//!
//! The committed fixture `tests/fixtures/linear_regression.run.jsonl` is rewritten from a fresh
//! recording by running [`observatory_record_fixture_current`] with
//! `MODELSPEC_REGENERATE_FIXTURES=1`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";
const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/linear_regression.run.jsonl";
const REGENERATE: &str = "MODELSPEC_REGENERATE_FIXTURES";
const WAIT: Duration = Duration::from_secs(120);

/// The fields of a `RunEvent` that differ per invocation (`RecordedRunFile`).
const PER_INVOCATION: [&str; 2] = ["training_run_id", "run_event_id"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// A fresh path `name` under the test temp dir, without a file at it.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("observatory_record");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn path_arg(path: &Path) -> String {
    path.display().to_string()
}

/// A finished `modelspec` process.
struct Finished {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: String,
}

/// Runs `modelspec <args>` from the repository root to its end, killing it after [`WAIT`].
fn modelspec(args: &[&str]) -> Finished {
    let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("modelspec starts");
    let mut out = child.stdout.take().expect("stdout is piped");
    let mut err = child.stderr.take().expect("stderr is piped");
    let stdout = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = out.read_to_end(&mut bytes);
        bytes
    });
    let stderr = std::thread::spawn(move || {
        let mut all = String::new();
        let _ = err.read_to_string(&mut all);
        all
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the status is read") {
            break status;
        }
        if start.elapsed() > WAIT {
            let _ = child.kill();
            let _ = child.wait();
            panic!("modelspec {args:?} still runs after {WAIT:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Finished {
        status,
        stdout: stdout.join().expect("stdout reader"),
        stderr: stderr.join().expect("stderr reader"),
    }
}

/// A serving `modelspec` process, killed when dropped.
struct Served {
    child: Child,
    address: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<String>>,
}

impl Served {
    /// Starts `modelspec <args>` and waits for the `serving http://<addr>/events` line on stderr.
    fn start(args: &[&str]) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(args)
            .current_dir(repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("modelspec starts");
        let mut out = child.stdout.take().expect("stdout is piped");
        let stdout = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = out.read_to_end(&mut bytes);
            bytes
        });
        let (lines, received) = mpsc::channel();
        let err = child.stderr.take().expect("stderr is piped");
        let stderr = std::thread::spawn(move || {
            let mut all = String::new();
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                all.push_str(&line);
                all.push('\n');
                let _ = lines.send(line);
            }
            all
        });
        let mut served = Served {
            child,
            address: String::new(),
            stdout: Some(stdout),
            stderr: Some(stderr),
        };
        loop {
            match received.recv_timeout(WAIT) {
                Ok(line) => {
                    if let Some(rest) = line.strip_prefix("serving http://") {
                        let address = rest
                            .strip_suffix("/events")
                            .unwrap_or_else(|| panic!("`serving http://<addr>/events`: {line}"));
                        served.address = address.to_owned();
                        return served;
                    }
                }
                Err(error) => {
                    let status = served.child.try_wait();
                    let stderr = served.stop().1;
                    panic!("no `serving` line on stderr ({error}, status {status:?}): {stderr}");
                }
            }
        }
    }

    /// Sends SIGTERM, waits for the process to exit and returns its status, stdout and stderr.
    fn terminate(&mut self) -> (ExitStatus, Vec<u8>, String) {
        let pid = self.child.id().to_string();
        let sent = Command::new("kill")
            .args(["-s", "TERM", &pid])
            .status()
            .expect("kill runs");
        assert!(sent.success(), "kill -s TERM {pid}: {sent}");
        let start = Instant::now();
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("the status is read") {
                break status;
            }
            assert!(
                start.elapsed() < WAIT,
                "the process still serves after SIGTERM"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        let (stdout, stderr) = self.stop();
        (status, stdout, stderr)
    }

    /// Kills the process and returns its stdout and stderr.
    fn stop(&mut self) -> (Vec<u8>, String) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let stdout = self
            .stdout
            .take()
            .map(|handle| handle.join().expect("stdout reader"))
            .unwrap_or_default();
        let stderr = self
            .stderr
            .take()
            .map(|handle| handle.join().expect("stderr reader"))
            .unwrap_or_default();
        (stdout, stderr)
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One Server-Sent Events frame as a client reads it, its `data:` line kept as sent.
#[derive(Debug, PartialEq)]
struct Frame {
    event: String,
    id: String,
    data: String,
}

/// The body of `GET /events` at `address`, read until the server closes, after checking the
/// status line and the content type.
fn read_stream(address: &str) -> Vec<Frame> {
    let mut stream = TcpStream::connect(address).expect("the server accepts a connection");
    stream.set_read_timeout(Some(WAIT)).expect("a read timeout");
    write!(
        stream,
        "GET /events HTTP/1.1\r\nHost: {address}\r\nAccept: text/event-stream\r\n\r\n"
    )
    .expect("the request is sent");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("the server ends the response");
    let (head, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("a response head: {response:?}"));
    let (status, headers) = head.split_once("\r\n").unwrap_or((head, ""));
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert!(
        headers
            .lines()
            .any(|h| h.eq_ignore_ascii_case("content-type: text/event-stream")),
        "{headers}"
    );
    frames(body)
}

/// The frames of an event-stream body: blocks separated by a blank line, each with exactly one
/// `event:`, `id:` and `data:` line.
fn frames(body: &str) -> Vec<Frame> {
    assert!(
        body.ends_with("\n\n"),
        "the stream ends with a whole frame: {body:?}"
    );
    body.trim_end_matches('\n')
        .split("\n\n")
        .map(|block| {
            let (mut event, mut id, mut data) = (None, None, None);
            for line in block.lines() {
                let (field, value) = line
                    .split_once(": ")
                    .unwrap_or_else(|| panic!("`<field>: <value>`: {line:?}"));
                let slot = match field {
                    "event" => &mut event,
                    "id" => &mut id,
                    "data" => &mut data,
                    other => panic!("unexpected field {other:?} in {block:?}"),
                };
                assert!(slot.is_none(), "one {field} line per frame: {block:?}");
                *slot = Some(value.to_owned());
            }
            Frame {
                event: event.unwrap_or_else(|| panic!("an event line: {block:?}")),
                id: id.unwrap_or_else(|| panic!("an id line: {block:?}")),
                data: data.unwrap_or_else(|| panic!("a data line: {block:?}")),
            }
        })
        .collect()
}

fn json(line: &str) -> Value {
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"))
}

/// The lines of a recorded run file, after checking that it ends with a newline.
fn recorded_lines(path: &Path) -> Vec<String> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("the recording {} is readable: {e}", path.display()));
    assert!(content.ends_with('\n'), "the recording ends with a newline");
    content.lines().map(str::to_owned).collect()
}

fn schema() -> jsonschema::Validator {
    let text = std::fs::read_to_string(repo_root().join(SCHEMA)).expect("the schema is committed");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

/// Every event validates against the schema, the sequences run 0, 1, 2, ..., one
/// `training_run_id` is shared by all, `RunStarted` is first and `RunFinished` last.
fn assert_conforms(events: &[Value]) {
    let validator = schema();
    assert!(!events.is_empty());
    for (k, event) in events.iter().enumerate() {
        let errors: Vec<String> = validator
            .iter_errors(event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "event {k} {event}: {errors:?}");
        assert_eq!(event["sequence"], k as u64, "{event}");
        assert_eq!(event["training_run_id"], events[0]["training_run_id"]);
    }
    assert_eq!(events[0]["body"]["kind"], "RunStarted");
    assert_eq!(events[events.len() - 1]["body"]["kind"], "RunFinished");
}

/// Records `examples/linear_regression.mspec` with `train --record` and no `--serve` into
/// `name`, checking that stdout is the run's without the flag.
fn record(name: &str) -> PathBuf {
    let path = scratch(name);
    let recorded = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--record",
        &path_arg(&path),
    ]);
    assert_eq!(recorded.status.code(), Some(0), "{}", recorded.stderr);
    assert!(recorded.stderr.is_empty(), "stderr: {}", recorded.stderr);
    let plain = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
    assert_eq!(
        text(&recorded.stdout),
        text(&plain.stdout),
        "stdout is the run's without --record"
    );
    path
}

/// `observatory.record.matches-live`: one invocation serves and records; the events in the file
/// equal, in order and field by field, the events a client read from the live stream.
#[test]
fn observatory_record_matches_live() {
    let path = scratch("live.run.jsonl");
    let mut served = Served::start(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--serve",
        "127.0.0.1:0",
        "--record",
        &path_arg(&path),
    ]);
    let live = read_stream(&served.address);
    let (status, stdout, _) = served.terminate();
    assert_eq!(status.code(), Some(0), "{status}");
    let plain = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
    assert_eq!(text(&stdout), text(&plain.stdout), "stdout is unchanged");
    let lines = recorded_lines(&path);
    let live_events: Vec<Value> = live.iter().map(|f| json(&f.data)).collect();
    let recorded: Vec<Value> = lines.iter().map(|l| json(l)).collect();
    assert_conforms(&recorded);
    assert_eq!(recorded.len(), 503, "RunStarted, 501 steps, RunFinished");
    assert_eq!(recorded, live_events);
    let data: Vec<&str> = live.iter().map(|f| f.data.as_str()).collect();
    assert_eq!(lines, data, "each line is the frame's data, byte for byte");
}

/// `observatory.replay.matches-recording`: replaying the file serves the same events, the same
/// `sequence` and the same `id:` lines, each valid against the RunEvent schema.
#[test]
fn observatory_replay_matches_recording() {
    let path = record("replayed.run.jsonl");
    let lines = recorded_lines(&path);
    let mut served = Served::start(&["observe", &path_arg(&path), "--serve", "127.0.0.1:0"]);
    let first = read_stream(&served.address);
    let second = read_stream(&served.address);
    let (status, stdout, stderr) = served.terminate();
    assert_eq!(
        status.code(),
        Some(0),
        "observe exits 0 on SIGTERM: {stderr}"
    );
    assert!(stdout.is_empty(), "stdout: {}", text(&stdout));
    assert_eq!(first, second, "a second client reads the same stream");
    let events: Vec<Value> = first.iter().map(|f| json(&f.data)).collect();
    assert_conforms(&events);
    assert_eq!(first.len(), lines.len());
    for (frame, line) in first.iter().zip(&lines) {
        let event = json(line);
        assert_eq!(frame.data, *line);
        assert_eq!(frame.id, event["sequence"].to_string());
        assert_eq!(frame.event, event["body"]["kind"].as_str().unwrap());
    }
}

/// `observe` without `--serve` serves on a free port of 127.0.0.1.
#[test]
fn observe_defaults_to_a_free_loopback_port() {
    let path = record("default_port.run.jsonl");
    let mut served = Served::start(&["observe", &path_arg(&path)]);
    assert!(
        served.address.starts_with("127.0.0.1:"),
        "{}",
        served.address
    );
    let frames = read_stream(&served.address);
    served.stop();
    assert_eq!(frames.len(), recorded_lines(&path).len());
}

/// `observatory.replay.refuses-damaged-file`: a truncated file, a file with a `sequence` gap and
/// a file mixing two `training_run_id` values are each refused with a one-line error and exit 1,
/// before anything is served.
#[test]
fn observatory_replay_refuses_damaged_file() {
    let path = record("whole.run.jsonl");
    let content = std::fs::read_to_string(&path).expect("the recording is readable");
    let lines: Vec<&str> = content.lines().collect();
    let joined = |lines: &[String]| lines.iter().map(|l| format!("{l}\n")).collect::<String>();
    let owned: Vec<String> = lines.iter().map(|l| (*l).to_owned()).collect();

    let cut_mid_line = content[..content.len() - 10].to_owned();
    let cut_at_line = joined(&owned[..owned.len() - 1]);
    let mut gap = owned.clone();
    gap.remove(5);
    let mut mixed = owned.clone();
    let mut other: Value = json(&mixed[7]);
    other["training_run_id"] = Value::String("00000000-0000-4000-8000-000000000000".into());
    mixed[7] = serde_json::to_string(&other).unwrap();
    assert!(
        schema().is_valid(&other),
        "the mixed line is a valid RunEvent"
    );

    for (name, damaged) in [
        ("truncated mid-line", cut_mid_line),
        ("truncated at a line", cut_at_line),
        ("sequence gap", joined(&gap)),
        ("two training_run_id values", joined(&mixed)),
    ] {
        let file = scratch(&format!("{}.run.jsonl", name.replace(' ', "_")));
        std::fs::write(&file, damaged).expect("the damaged file is written");
        let refused = modelspec(&["observe", &path_arg(&file), "--serve", "127.0.0.1:0"]);
        assert_eq!(refused.status.code(), Some(1), "{name}: {}", refused.stderr);
        assert!(refused.stdout.is_empty(), "{name}: stdout");
        assert_eq!(
            refused.stderr.lines().count(),
            1,
            "{name}: one line: {}",
            refused.stderr
        );
        assert!(refused.stderr.ends_with('\n'), "{name}: {}", refused.stderr);
        assert!(
            !refused.stderr.contains("serving"),
            "{name}: nothing is served: {}",
            refused.stderr
        );
    }
}

/// Each event of `lines` without the per-invocation fields.
fn invariant(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .map(|line| {
            let mut event = json(line);
            let object = event.as_object_mut().expect("an event is an object");
            for field in PER_INVOCATION {
                assert!(object.remove(field).is_some(), "{field} in {line}");
            }
            event
        })
        .collect()
}

/// `observatory.record.fixture-current`: the committed linear-regression fixture equals a fresh
/// recording except for the per-invocation fields, `training_run_id` and `run_event_id`.
#[test]
fn observatory_record_fixture_current() {
    let path = record("fixture.run.jsonl");
    let fixture = repo_root().join(FIXTURE);
    if std::env::var_os(REGENERATE).is_some() {
        std::fs::copy(&path, &fixture).expect("the fixture is written");
    }
    let fresh = recorded_lines(&path);
    let committed = recorded_lines(&fixture);
    assert_conforms(&committed.iter().map(|l| json(l)).collect::<Vec<_>>());
    assert_eq!(
        invariant(&committed),
        invariant(&fresh),
        "the fixture is a recording of {EXAMPLE}; rerun this test with {REGENERATE}=1"
    );
}

/// A file that cannot be created is refused before the run, on one line, with exit 1.
#[test]
fn a_record_path_that_cannot_be_created_is_refused() {
    let missing = scratch("no_such_dir").join("run.jsonl");
    let refused = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--record",
        &path_arg(&missing),
    ]);
    assert_eq!(refused.status.code(), Some(1), "{}", refused.stderr);
    assert!(refused.stdout.is_empty(), "no step is run");
    assert_eq!(refused.stderr.lines().count(), 1, "{}", refused.stderr);
}
