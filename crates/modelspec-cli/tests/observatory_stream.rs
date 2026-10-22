//! `modelspec train --backend cpu --serve <addr>`: the live event stream of a training run
//! (story:observatory-live-event-stream; `spec/domains/observatory.yaml`, `RunEvent`,
//! `RunEventBody`, `ServeAddress`, `StreamFrame`).
//!
//! Each test starts the binary with `--serve 127.0.0.1:0`, reads the bound address from the
//! `serving http://<addr>/events` line on stderr, reads `GET /events` to its end as a client
//! would, and stops the process, which serves until it is interrupted. Every event is checked
//! against the JSON Schema `ess generate --kind schema` wrote for `modelspec.observatory.RunEvent`
//! under `crates/modelspec-observatory/schema/`; nothing here reads the Rust types of the stream.
//!
//! Scenario ids of the story map to the tests: `observatory.stream.linear-regression-events-conform`
//! is [`linear_regression_events_conform`], `observatory.stream.final-loss-equals-cli` is
//! [`final_loss_equals_cli`], and `observatory.stream.lifecycle-order` is [`lifecycle_order`] and
//! [`a_data_refusal_ends_the_stream_with_run_failed`].

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";
const WAIT: Duration = Duration::from_secs(120);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// Runs `modelspec <args>` from the repository root to its end.
fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

/// Writes `source` to a fresh file named `name` under the test temp dir and returns its path.
fn source_file(name: &str, source: &str) -> String {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("observatory_stream");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, source).expect("the source is written");
    path.display().to_string()
}

/// A serving `modelspec train` process, killed when dropped.
struct Served {
    child: Child,
    address: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<String>>,
}

impl Served {
    /// Starts `modelspec train <path> --backend cpu --serve 127.0.0.1:0` and waits for the line
    /// naming the bound address.
    fn start(path: &str) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(["train", path, "--backend", "cpu", "--serve", "127.0.0.1:0"])
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

/// One Server-Sent Events frame as a client reads it.
#[derive(Debug)]
struct Frame {
    event: String,
    id: String,
    data: Value,
}

/// The status line, headers and body of `GET <path>` at `address`, read until the server closes.
fn get(address: &str, path: &str) -> (String, String, String) {
    let mut stream = TcpStream::connect(address).expect("the server accepts a connection");
    stream.set_read_timeout(Some(WAIT)).expect("a read timeout");
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {address}\r\nAccept: text/event-stream\r\n\r\n"
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
    (status.to_owned(), headers.to_owned(), body.to_owned())
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
                data: serde_json::from_str(
                    &data.unwrap_or_else(|| panic!("a data line: {block:?}")),
                )
                .expect("data is JSON"),
            }
        })
        .collect()
}

/// Reads the whole stream of a serving process.
fn read_stream(served: &Served) -> Vec<Frame> {
    let (status, headers, body) = get(&served.address, "/events");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert!(
        headers
            .lines()
            .any(|h| h.eq_ignore_ascii_case("content-type: text/event-stream")),
        "{headers}"
    );
    frames(&body)
}

fn schema() -> jsonschema::Validator {
    let text = std::fs::read_to_string(repo_root().join(SCHEMA)).expect("the schema is committed");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

/// Every frame's data validates against the schema, its `event:` is the body's kind, its `id:`
/// its sequence, the sequences run 0, 1, 2, ... and one `training_run_id` is shared by all.
fn assert_conforms(frames: &[Frame]) {
    let validator = schema();
    assert!(!frames.is_empty());
    for (k, frame) in frames.iter().enumerate() {
        let errors: Vec<String> = validator
            .iter_errors(&frame.data)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "event {k} {}: {errors:?}", frame.data);
        assert_eq!(frame.data["sequence"], k as u64, "{}", frame.data);
        assert_eq!(frame.id, k.to_string());
        assert_eq!(frame.event, frame.data["body"]["kind"].as_str().unwrap());
    }
    let run = &frames[0].data["training_run_id"];
    assert!(frames.iter().all(|f| &f.data["training_run_id"] == run));
}

fn kinds(frames: &[Frame]) -> Vec<&str> {
    frames
        .iter()
        .map(|f| f.data["body"]["kind"].as_str().unwrap())
        .collect()
}

fn steps(frames: &[Frame]) -> Vec<&Value> {
    frames
        .iter()
        .filter(|f| f.data["body"]["kind"] == "StepCompleted")
        .map(|f| &f.data["body"]["value"])
        .collect()
}

#[test]
fn linear_regression_events_conform() {
    let mut served = Served::start(EXAMPLE);
    let frames = read_stream(&served);
    served.stop();
    assert_conforms(&frames);
    let steps = steps(&frames);
    assert_eq!(
        steps.len(),
        501,
        "one StepCompleted per loss, steps 0..=500"
    );
    for (k, step) in steps.iter().enumerate() {
        assert_eq!(step["step"], k as u64);
        assert_eq!(step["metrics"], Value::Array(vec![]));
        assert_eq!(step["parameters"], Value::Array(vec![]));
    }
}

#[test]
fn final_loss_equals_cli() {
    let mut served = Served::start(EXAMPLE);
    let frames = read_stream(&served);
    let (stdout, _) = served.stop();
    let stdout = text(&stdout);
    let plain = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
    assert_eq!(plain.status.code(), Some(0));
    assert_eq!(
        stdout,
        text(&plain.stdout),
        "stdout is the run's without --serve"
    );
    let printed = stdout
        .lines()
        .find_map(|line| line.strip_prefix("step 500 loss "))
        .unwrap_or_else(|| panic!("a `step 500 loss <v>` line: {stdout}"));
    let last = steps(&frames).last().copied().expect("a StepCompleted");
    assert_eq!(last["step"], 500);
    let loss = last["loss"].as_f64().expect("a number") as f32;
    assert_eq!(loss.to_string(), printed);
}

#[test]
fn lifecycle_order() {
    let mut served = Served::start(EXAMPLE);
    let frames = read_stream(&served);
    served.stop();
    let kinds = kinds(&frames);
    assert_eq!(kinds.first(), Some(&"RunStarted"));
    assert_eq!(kinds.last(), Some(&"RunFinished"));
    assert!(
        kinds[1..kinds.len() - 1]
            .iter()
            .all(|k| *k == "StepCompleted"),
        "{kinds:?}"
    );
    let started = &frames[0].data["body"]["value"];
    assert_eq!(started["steps"], 500);
    assert_eq!(started["start_step"], 0);
}

const UNREAD_X: &str = "model Const {
    param b: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return b;
    }
}

train Fit {
    model = Const;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
";

#[test]
fn a_data_refusal_ends_the_stream_with_run_failed() {
    let path = source_file("unread_x.mspec", UNREAD_X);
    let mut served = Served::start(&path);
    let frames = read_stream(&served);
    let (stdout, stderr) = served.stop();
    assert_conforms(&frames);
    assert_eq!(kinds(&frames), ["RunStarted", "RunFailed"]);
    let failed = &frames[1].data["body"]["value"];
    assert_eq!(failed["code"], "E-DATA-003");
    assert_eq!(failed["message"], "the input `x` is not read by the loss");
    assert!(stdout.is_empty(), "stdout: {}", text(&stdout));
    assert!(
        stderr.contains("E-DATA-003: the input `x` is not read by the loss\n"),
        "{stderr}"
    );
}

/// Sends `signal` (`TERM`, `INT`) to the serving process and waits for it to exit.
fn interrupt(served: &mut Served, signal: &str) -> std::process::ExitStatus {
    let pid = served.child.id().to_string();
    let sent = Command::new("kill")
        .args(["-s", signal, &pid])
        .status()
        .expect("kill runs");
    assert!(sent.success(), "kill -s {signal} {pid}: {sent}");
    let start = std::time::Instant::now();
    loop {
        if let Some(status) = served.child.try_wait().expect("the status is read") {
            return status;
        }
        assert!(
            start.elapsed() < WAIT,
            "the process still serves after SIG{signal}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// `StreamFrame` (wave observatory-1 coordinator, 2026-10-21): after the run the process keeps
/// serving, and on SIGINT or SIGTERM it stops and exits with the run's own status, 0 for a run
/// that finished and 1 for one that failed, as `modelspec train` without the flag does.
#[test]
fn an_interrupted_server_exits_with_the_runs_status() {
    let refused = source_file("unread_x_interrupted.mspec", UNREAD_X);
    for (path, last, code) in [
        (EXAMPLE, "RunFinished", 0),
        (refused.as_str(), "RunFailed", 1),
    ] {
        for signal in ["TERM", "INT"] {
            let mut served = Served::start(path);
            let frames = read_stream(&served);
            assert_eq!(kinds(&frames).last(), Some(&last), "{path}");
            let status = interrupt(&mut served, signal);
            let (stdout, _) = served.stop();
            assert_eq!(status.code(), Some(code), "{path} on SIG{signal}: {status}");
            if code == 0 {
                let plain = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
                assert_eq!(text(&stdout), text(&plain.stdout), "SIG{signal}");
            }
        }
    }
}

/// Starts a serving run of `path`, reads stderr (`watch_stderr`) or stdout line by line until `end`
/// says the lines read so far call for the signal, sends SIGTERM at once and returns the exit code
/// (128 + the signal for a process a signal killed).
fn terminate_on_line(path: &str, watch_stderr: bool, end: &dyn Fn(&[String]) -> bool) -> i32 {
    let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", path, "--backend", "cpu", "--serve", "127.0.0.1:0"])
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("modelspec starts");
    let out = child.stdout.take().expect("stdout is piped");
    let err = child.stderr.take().expect("stderr is piped");
    let (watched, other): (Box<dyn Read + Send>, Box<dyn Read + Send>) = if watch_stderr {
        (Box::new(err), Box::new(out))
    } else {
        (Box::new(out), Box::new(err))
    };
    let drain = std::thread::spawn(move || {
        let mut other = other;
        let _ = std::io::copy(&mut other, &mut std::io::sink());
    });
    let mut seen = Vec::new();
    for line in BufReader::new(watched).lines() {
        let Ok(line) = line else { break };
        seen.push(line);
        if end(&seen) {
            let pid = Pid::from_raw(i32::try_from(child.id()).expect("a pid"));
            kill(pid, Signal::SIGTERM).expect("SIGTERM is sent");
            break;
        }
    }
    let start = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the status is read") {
            break status;
        }
        if start.elapsed() > WAIT {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the process still serves after SIGTERM: {seen:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let _ = drain.join();
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().expect("a code or a signal"))
}

/// `StreamFrame` (wave observatory-1 coordinator, 2026-10-22): the run has ended once it has
/// printed its last stdout line, or for a failed run its failure on stderr, so a SIGTERM sent the
/// moment that line is read exits with the run's status, 0 or 1, every time and never 143.
#[test]
fn a_signal_on_the_last_line_exits_with_the_runs_status() {
    const RUNS: usize = 100;
    let last_param = |seen: &[String]| {
        let inits = seen.iter().filter(|l| l.starts_with("init @")).count();
        let params = seen.iter().filter(|l| l.starts_with("param @")).count();
        inits > 0 && params == inits
    };
    let refused = source_file("unread_x_last_line.mspec", UNREAD_X);
    let failure_line = |seen: &[String]| seen.last().is_some_and(|l| l.starts_with("E-DATA-003:"));
    let mut wrong = Vec::new();
    for (path, watch_stderr, end, code) in [
        (EXAMPLE, false, &last_param as &dyn Fn(&[String]) -> bool, 0),
        (refused.as_str(), true, &failure_line, 1),
    ] {
        let mut codes = BTreeMap::new();
        for _ in 0..RUNS {
            *codes
                .entry(terminate_on_line(path, watch_stderr, end))
                .or_insert(0usize) += 1;
        }
        if codes.keys().any(|c| *c != code) {
            wrong.push(format!(
                "{path}: {RUNS} runs, expected exit {code} every run, got {codes:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `StreamFrame`: "a signal before the run ends stops the process as it stops a run without the
/// flag". A run of 200000 steps signalled on its `serving` line is killed by SIGTERM.
#[test]
fn a_signal_before_the_run_ends_kills_the_process() {
    let long = std::fs::read_to_string(repo_root().join(EXAMPLE))
        .expect("the example is readable")
        .replace("steps = 500;", "steps = 200000;");
    let path = source_file("long_run.mspec", &long);
    let serving = |seen: &[String]| seen.last().is_some_and(|l| l.starts_with("serving "));
    assert_eq!(terminate_on_line(&path, true, &serving), 128 + 15);
}

#[test]
fn a_late_client_reads_the_same_stream() {
    let mut served = Served::start(EXAMPLE);
    let first = read_stream(&served);
    let second = read_stream(&served);
    served.stop();
    let data = |frames: &[Frame]| frames.iter().map(|f| f.data.clone()).collect::<Vec<_>>();
    assert_eq!(data(&first), data(&second));
}

#[test]
fn another_path_is_not_found() {
    let mut served = Served::start(EXAMPLE);
    let (status, _, _) = get(&served.address, "/");
    served.stop();
    assert_eq!(status, "HTTP/1.1 404 Not Found");
}

#[test]
fn a_non_loopback_address_is_a_usage_error() {
    for address in ["0.0.0.0:0", "192.0.2.1:8080", "[::]:0", "example.org:80"] {
        let output = modelspec(&["train", EXAMPLE, "--backend", "cpu", "--serve", address]);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{address}: {}",
            text(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{address}");
    }
}

#[test]
fn a_run_without_the_flag_serves_nothing() {
    let output = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
}
