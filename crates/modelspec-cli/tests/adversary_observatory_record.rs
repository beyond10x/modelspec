//! Adversary pass on story:observatory-run-recording-replay (`spec/domains/observatory.yaml`,
//! `RecordedRunFile`, `RunEventBody`): `train --record` and `observe` driven from the contract the
//! unit wrote.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const WAIT: Duration = Duration::from_secs(120);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_observatory_record");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn arg(path: &Path) -> String {
    path.display().to_string()
}

struct Finished {
    status: Option<ExitStatus>,
    stdout: String,
    stderr: String,
}

/// Runs `modelspec <args>` until it exits or `limit` passes; a process still running then is
/// killed, reaped and reported with no status.
fn modelspec(args: &[&str], limit: Duration) -> Finished {
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
        let mut all = String::new();
        let _ = out.read_to_string(&mut all);
        all
    });
    let stderr = std::thread::spawn(move || {
        let mut all = String::new();
        let _ = err.read_to_string(&mut all);
        all
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the status is read") {
            break Some(status);
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Finished {
        status,
        stdout: stdout.join().expect("stdout reader"),
        stderr: stderr.join().expect("stderr reader"),
    }
}

/// The example with `sgd(lr = 100.0)`, whose loss overflows (unit 2's divergent fixture).
fn divergent() -> PathBuf {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example");
    let changed = source.replace("sgd(lr = 0.1)", "sgd(lr = 100.0)");
    assert_ne!(changed, source);
    let path = scratch("divergent.mspec");
    std::fs::write(&path, changed).expect("the source is written");
    path
}

/// A served `observe` process: its address, killed and reaped when dropped.
struct Observed {
    child: std::process::Child,
    address: String,
}

impl Drop for Observed {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn observe(path: &Path) -> Observed {
    let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["observe", &arg(path), "--serve", "127.0.0.1:0"])
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("modelspec starts");
    let err = child.stderr.take().expect("stderr is piped");
    let mut line = String::new();
    let mut reader = std::io::BufReader::new(err);
    std::io::BufRead::read_line(&mut reader, &mut line).expect("a stderr line");
    let mut observed = Observed {
        child,
        address: String::new(),
    };
    let address = line
        .trim_end()
        .strip_prefix("serving http://")
        .and_then(|rest| rest.strip_suffix("/events"))
        .unwrap_or_else(|| panic!("a serving line: {line:?}"));
    observed.address = address.to_owned();
    observed
}

/// The `data:` lines of the whole `GET /events` body at `address`.
fn data_lines(address: &str) -> Vec<String> {
    let mut stream = TcpStream::connect(address).expect("a connection");
    stream.set_read_timeout(Some(WAIT)).expect("a timeout");
    write!(stream, "GET /events HTTP/1.1\r\nHost: {address}\r\n\r\n").expect("the request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("the response ends");
    let (_, body) = response.split_once("\r\n\r\n").expect("a head");
    body.lines()
        .filter_map(|l| l.strip_prefix("data: ").map(str::to_owned))
        .collect()
}

/// `RecordedRunFile` and train.rs: "A run whose events cannot all be written reports that on one
/// line on stderr and exits 1". A full device (`/dev/full`, every write `ENOSPC`) is the disk
/// that fills during a run.
#[test]
fn adversary_a_record_write_failure_is_reported_on_one_line() {
    let run = modelspec(
        &[
            "train",
            EXAMPLE,
            "--backend",
            "cpu",
            "--record",
            "/dev/full",
        ],
        WAIT,
    );
    assert_eq!(run.status.and_then(|s| s.code()), Some(1), "{}", run.stderr);
    let lines: Vec<&str> = run
        .stderr
        .lines()
        .filter(|l| l.contains("cannot record"))
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "one line reports the record failure: {}",
        run.stderr
    );
}

/// `train --serve <busy addr> --record <existing file>`: the address is refused (exit 1), and a
/// recording already at the path, such as the previous run's, is not destroyed by a run that
/// never started.
#[test]
fn adversary_a_refused_serve_leaves_an_existing_recording_alone() {
    let busy = TcpListener::bind("127.0.0.1:0").expect("a port");
    let address = busy.local_addr().expect("its address").to_string();
    let path = scratch("previous.run.jsonl");
    std::fs::write(&path, "previous recording\n").expect("the file is written");
    let run = modelspec(
        &[
            "train",
            EXAMPLE,
            "--backend",
            "cpu",
            "--serve",
            &address,
            "--record",
            &arg(&path),
        ],
        WAIT,
    );
    drop(busy);
    assert_eq!(run.status.and_then(|s| s.code()), Some(1), "{}", run.stderr);
    assert!(run.stdout.is_empty(), "no step runs");
    assert_eq!(
        std::fs::read_to_string(&path).ok().as_deref(),
        Some("previous recording\n"),
        "the existing file at --record survives a refused --serve: {}",
        run.stderr
    );
}

fn event(sequence: u64, body: &str) -> String {
    format!(
        "{{\"run_event_id\":\"00000000-0000-4000-8000-{sequence:012}\",\"training_run_id\":\
         \"11111111-1111-4111-8111-111111111111\",\"sequence\":{sequence},\"body\":{body},\
         \"state\":\"Emitted\"}}\n"
    )
}

/// `RunEventBody` (spec/domains/observatory.yaml): "RunStarted first, then one StepCompleted per
/// recorded loss, then exactly one of RunFinished or RunFailed". A second `RunStarted` is no
/// stream a run sends, so `observe` refuses it before serving (exit 1, one line).
#[test]
fn adversary_observe_refuses_a_second_run_started() {
    let started = r#"{"kind":"RunStarted","value":{"steps":1,"start_step":0}}"#;
    let content = [
        event(0, started),
        event(1, started),
        event(2, r#"{"kind":"RunFinished"}"#),
    ]
    .concat();
    let path = scratch("two_starts.run.jsonl");
    std::fs::write(&path, content).expect("the file is written");
    let run = modelspec(
        &["observe", &arg(&path), "--serve", "127.0.0.1:0"],
        Duration::from_secs(10),
    );
    assert_eq!(
        run.status.and_then(|s| s.code()),
        Some(1),
        "refused, not served: {}",
        run.stderr
    );
    assert_eq!(run.stderr.lines().count(), 1, "{}", run.stderr);
}

/// unit 2 (E-TRAIN-001) recorded and replayed: the file ends `RunFailed` with the code after a
/// `StepCompleted` whose loss is `null`, and the replay serves those lines byte for byte.
#[test]
fn adversary_a_divergent_run_records_and_replays() {
    let source = divergent();
    let path = scratch("divergent.run.jsonl");
    let run = modelspec(
        &[
            "train",
            &arg(&source),
            "--backend",
            "cpu",
            "--record",
            &arg(&path),
        ],
        WAIT,
    );
    assert_eq!(run.status.and_then(|s| s.code()), Some(1), "{}", run.stderr);
    let content = std::fs::read_to_string(&path).expect("the recording");
    let lines: Vec<String> = content.lines().map(str::to_owned).collect();
    let last: Value = serde_json::from_str(&lines[lines.len() - 1]).expect("JSON");
    assert_eq!(last["body"]["kind"], "RunFailed");
    assert_eq!(last["body"]["value"]["code"], "E-TRAIN-001");
    let before: Value = serde_json::from_str(&lines[lines.len() - 2]).expect("JSON");
    assert_eq!(before["body"]["value"]["loss"], Value::Null);
    let observed = observe(&path);
    assert_eq!(data_lines(&observed.address), lines);
    assert_eq!(data_lines(&observed.address), lines, "a second client");
}
