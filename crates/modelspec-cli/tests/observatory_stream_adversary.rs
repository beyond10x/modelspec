//! Adversary cases against `modelspec train --serve` (story:observatory-live-event-stream),
//! driven from `spec/domains/observatory.yaml` (`StreamFrame`, `ServeAddress`, `RunEventBody`)
//! and the committed `RunEvent` JSON Schema, not from the Rust types of the stream.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";
const WAIT: Duration = Duration::from_secs(120);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source_file(name: &str, source: &str) -> String {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("observatory_stream_adversary");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, source).unwrap();
    path.display().to_string()
}

struct Served {
    child: Child,
    address: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
}

impl Served {
    fn start(path: &str) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(["train", path, "--backend", "cpu", "--serve", "127.0.0.1:0"])
            .current_dir(repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut out = child.stdout.take().unwrap();
        let stdout = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = out.read_to_end(&mut bytes);
            bytes
        });
        let (lines, received) = mpsc::channel();
        let err = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                let _ = lines.send(line);
            }
        });
        let mut served = Served {
            child,
            address: String::new(),
            stdout: Some(stdout),
        };
        loop {
            let line = received.recv_timeout(WAIT).expect("a `serving` line");
            if let Some(rest) = line.strip_prefix("serving http://") {
                served.address = rest.strip_suffix("/events").unwrap().to_owned();
                return served;
            }
        }
    }

    fn stop(&mut self) -> Vec<u8> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.stdout
            .take()
            .map(|h| h.join().unwrap())
            .unwrap_or_default()
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The body of `GET /events` with `extra` header lines, read until the server closes.
fn body(address: &str, extra: &str) -> String {
    let mut stream = TcpStream::connect(address).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    write!(
        stream,
        "GET /events HTTP/1.1\r\nHost: {address}\r\n{extra}\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("the server ends the response");
    let (head, body) = response.split_once("\r\n\r\n").expect("a head");
    assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
    body.to_owned()
}

/// `(id, data)` of each frame.
fn frames(body: &str) -> Vec<(u64, Value)> {
    if body.is_empty() {
        return Vec::new();
    }
    body.trim_end_matches('\n')
        .split("\n\n")
        .map(|block| {
            let id = block
                .lines()
                .find_map(|l| l.strip_prefix("id: "))
                .unwrap()
                .parse()
                .unwrap();
            let data = block
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap();
            (id, serde_json::from_str(data).expect("data is JSON"))
        })
        .collect()
}

fn validator() -> jsonschema::Validator {
    let text = std::fs::read_to_string(repo_root().join(SCHEMA)).unwrap();
    jsonschema::validator_for(&serde_json::from_str(&text).unwrap()).unwrap()
}

const DIVERGING: &str = "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 1000.0);
    steps = 50;
    seed = 42;
    device = cpu;
}
";

/// `StreamFrame.data`: "The JSON of one RunEvent on one line, valid against the JSON Schema
/// `ess generate --kind schema` writes for it." A run whose learning rate diverges is a valid
/// `.mspec` file that `modelspec train` accepts and runs to its end.
#[test]
fn a_diverging_run_streams_only_schema_valid_events() {
    let path = source_file("diverging.mspec", DIVERGING);
    let mut served = Served::start(&path);
    let frames = frames(&body(&served.address, ""));
    let stdout = String::from_utf8(served.stop()).unwrap();
    let validator = validator();
    let invalid: Vec<String> = frames
        .iter()
        .filter(|(_, data)| !validator.is_valid(data))
        .map(|(id, data)| format!("{id}: {}", data["body"]))
        .take(3)
        .collect();
    let last_lines: Vec<&str> = stdout.lines().rev().take(2).collect();
    assert!(
        invalid.is_empty(),
        "{} of {} events fail the schema, first: {invalid:?}; stdout ends {last_lines:?}",
        frames
            .iter()
            .filter(|(_, d)| !validator.is_valid(d))
            .count(),
        frames.len()
    );
}

/// `StreamFrame`: "from the one after a `Last-Event-ID` request header".
#[test]
fn last_event_id_boundaries() {
    let mut served = Served::start(EXAMPLE);
    let all = frames(&body(&served.address, ""));
    let last = all.last().unwrap().0;
    let after_0 = frames(&body(&served.address, "Last-Event-ID: 0\r\n"));
    let after_last = body(&served.address, &format!("Last-Event-ID: {last}\r\n"));
    let past_last = body(
        &served.address,
        &format!("Last-Event-ID: {}\r\n", last + 10),
    );
    let huge = body(&served.address, "Last-Event-ID: 18446744073709551615\r\n");
    served.stop();
    assert_eq!(after_0.first().map(|f| f.0), Some(1));
    assert_eq!(after_0.len(), all.len() - 1);
    assert_eq!(after_last, "");
    assert_eq!(past_last, "");
    assert_eq!(huge, "");
}

/// Two clients connected before the run ends each read the whole stream, and a client that
/// drops mid-stream does not stop the server for the next one.
#[test]
fn concurrent_and_disconnecting_clients() {
    let mut served = Served::start(EXAMPLE);
    let address = served.address.clone();
    let mut dropped = TcpStream::connect(&address).unwrap();
    write!(dropped, "GET /events HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut first = [0u8; 64];
    let _ = dropped.read(&mut first).unwrap();
    drop(dropped);
    let a = std::thread::spawn({
        let address = address.clone();
        move || body(&address, "")
    });
    let b = std::thread::spawn({
        let address = address.clone();
        move || body(&address, "")
    });
    let (a, b) = (a.join().unwrap(), b.join().unwrap());
    let late = body(&address, "");
    served.stop();
    assert_eq!(a, b);
    assert_eq!(a, late);
    let ids: Vec<u64> = frames(&a).iter().map(|f| f.0).collect();
    assert_eq!(ids, (0..ids.len() as u64).collect::<Vec<_>>());
    assert_eq!(frames(&a).last().unwrap().1["body"]["kind"], "RunFinished");
}

/// An address that is a valid loopback `ServeAddress` but cannot be bound is a runtime error:
/// exit 1 (AGENTS.md exit codes), nothing on stdout, no run.
#[test]
fn a_port_in_use_exits_1_without_running() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = taken.local_addr().unwrap().to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", EXAMPLE, "--backend", "cpu", "--serve", &address])
        .current_dir(repo_root())
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(output.stdout.is_empty());
    assert!(stderr.contains("cannot serve on"), "{stderr}");
}

/// A port that is not a `u16` is a usage error (exit 2), for both address forms.
#[test]
fn an_unparsable_serve_address_is_a_usage_error() {
    for address in [
        "localhost:99999",
        "127.0.0.1:99999",
        "localhost:",
        "127.0.0.1:-1",
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(["train", EXAMPLE, "--backend", "cpu", "--serve", address])
            .current_dir(repo_root())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{address}");
        assert!(output.stdout.is_empty(), "{address}");
    }
}
